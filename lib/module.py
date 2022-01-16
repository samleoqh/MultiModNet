import math
import os
import torch
from torch import nn, einsum
from torchvision import models
import torch.nn.functional as F
from einops import rearrange, repeat

from lib.backbone import *
from pretrainedmodels import se_resnext50_32x4d, resnet50, resnet34

pre_trained_backbone = '../ckpt/pre_trained_backbone'
mobilenet_path = os.path.join(pre_trained_backbone, 'mobilenet', 'mobilenetv3-large-1cd25616.pth' )

def weight_xavier_init(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                # nn.init.orthogonal_(module.weight)
                # nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class BatchNorm1d_GNN(nn.BatchNorm1d):
    '''Batch normalization over features'''

    def __init__(self, num_features):
        super(BatchNorm1d_GNN, self).__init__(num_features)

    def forward(self, x):
        return super(BatchNorm1d_GNN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class Encoder_AE(nn.Module):
    def __init__(self,
                 in_dim=None,
                 out_dim=16,
                 heads=1,
                 mv=3
                 ):
        super(Encoder_AE, self).__init__()
        if in_dim is None:
            in_dim = {'q': 256, 'z': 512, 'k': 1024}
        self.multi_heads = heads
        self.cross_views = mv
        self.label_coef = nn.Parameter(torch.eye(out_dim))

        self.k = nn.Conv2d(in_dim['k'], out_dim, 3, padding=1, bias=True)
        self.z = nn.Conv2d(in_dim['z'], out_dim * heads, 3, padding=1, bias=True)
        self.q = nn.Conv2d(in_dim['q'], out_dim, 3, padding=1, bias=True)

    def forward(self, x):
        b, _, kh, kw = x['k'].size()
        b, _, zh, zw = x['z'].size()
        b, _, qh, qw = x['q'].size()  # qh = 2*zh = 4*kh

        m = self.multi_heads
        cv = int(self.cross_views)

        K = [repeat(self.view_aug(self.k(self.view_aug(x['k'], i, False)), i, True).reshape(b, 1, 1, kh * kw, -1),
                    'b v c i j -> b v (y c) i j', y=m)
             for i in range(cv)]
        K = torch.cat(K, 1)  # b, 4, m kh*kw -1

        Q = self.q(x['q'])
        Z = self.z(x['z'])

        b, v, m, n, d = K.size()
        Ks = K.reshape(b, n, v, m * d).reshape(b, v * m * d, kh, kw)

        Zs = F.interpolate(Z, (qh, qw), mode='bilinear', align_corners=False)
        Ks = F.interpolate(Ks, (qh, qw), mode='bilinear', align_corners=False)
        QZK = torch.cat((Q, Zs, Ks), 1)

        Q = Q.reshape(b, 1, 1, qh * qw, -1)
        Q = repeat(Q, 'b v c i j -> b (x v) (y c) i j', x=cv, y=m)  # b m 4 hw -1, v=1, c=1
        Z = Z.reshape(b, 1, m, zh * zw, -1)
        Z = repeat(Z, 'b v m i j -> b (x v) m i j', x=cv)  # b 4 m hw -1, v=m

        t = lambda tensor: rearrange(tensor, 'b v m n d -> b v m d n')

        A = Q @ (torch.tanh(t(Z) @ Z) + self.label_coef) @ t(K)  # b, v, m, qh*qw x kh*hw
        A = torch.relu(torch.sum(torch.tanh(A), 1).sum(1))

        return A, QZK  # b, out_dim*[(mv+1)heads + 1], qh, qw

    @staticmethod
    def view_aug(x, v=0, r=False):
        assert 0 <= v <= 2
        "v shall be in range of 0-4"

        if v == 1:
            return x.flip(3)  # flip vertically
        elif v == 2:
            # return x.flip(3)  # flip vertically
            return x.permute(0, 1, 3, 2)  # transpose
        # elif v == 3:
        #     return x.transpose(2, 3).flip(3 if not r else 2)  # rotation 270
        else:
            return x


class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, bnorm=True,
                 activation=None, dropout=None, adj=True):
        super(GCN_Layer, self).__init__()
        self.bnorm = bnorm
        self.adj = adj
        fc = [nn.Linear(in_features, out_features)]
        if bnorm:
            fc.append(BatchNorm1d_GNN(out_features))
        if activation is not None:
            fc.append(activation)
        if dropout is not None:
            fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x, A, is_norm = data

        if not is_norm:
            max_coef = torch.sum(A, dim=2) + 1.e-7
            A = A/max_coef.unsqueeze(dim=2)
            # A = torch.softmax(A, 2)
            # A = torch.tanh(A)

        y = self.fc(torch.bmm(A, x))

        return [y, A, True]


class PAF_block(nn.Module):
    def __init__(self,
                 in_ch,
                 hidden_ch,
                 kernel=6,
                 heads=3,
                 m_views=4,
                 depth=1,
                 dropout=0.,
                 norm=True,
                 activation=None,
                 shortcut=True,
                 eps=2e-8):
        super(PAF_block, self).__init__()
        self.eps = eps
        if in_ch is None:
            self.in_ch = {'q': 256, 'z': 512, 'k': 1024}
        else:
            self.in_ch = in_ch

        if activation is None:
            self.activation = nn.PReLU()
        else:
            self.activation = activation
        self.sum = shortcut
        self.kernel = kernel
        self.heads = heads
        self.qzk_A = Encoder_AE(self.in_ch, self.kernel, heads, m_views)
        self.rs = nn.Sequential(
            nn.Conv2d(self.kernel * (heads * (m_views + 1) + 1),
                      hidden_ch,
                      kernel_size=3, stride=1, padding=1, bias=False),
            self.activation,
            nn.BatchNorm2d(hidden_ch, momentum=0.01),
            nn.Dropout2d(dropout),
        )

        self.gcn_k = nn.ModuleList([])
        for i in range(depth):
            in_feat = self.in_ch['k'] if i == 0 else hidden_ch
            adj = True if i == 0 else False
            self.gcn_k.append(
                GCN_Layer(in_features=in_feat, out_features=hidden_ch, bnorm=norm,
                          activation=self.activation, dropout=dropout, adj=adj)
            )

    def forward(self, x):
        b, _, hh, ww = x['q'].size()
        _, ch, kh, kw = x['k'].size()
        _, cz, zh, zw = x['z'].size()
        # ENCoder
        A, qzk= self.qzk_A(x)  # b n md, b v m n d

        gx = x['k'].reshape(b, -1, ch)
        lap_norm = False

        for s, gcn in enumerate(self.gcn_k):
            if s==0:
                gx, A, lap_norm = gcn((gx, A, lap_norm))
            else:
                gx = F.interpolate(gx.reshape(b, -1, hh, ww), (kh, kw), mode='bilinear', align_corners=False)
                gx, A, lap_norm = gcn((gx.reshape(b, kh*kw, -1), A, lap_norm))


        if self.sum:
            return gx.reshape(b, -1, hh, ww) + self.rs(qzk) , A, 0
        else:
            return gx.reshape(b, -1, hh, ww), A, 0





