
from lib.module import *

def load_model(name='HPAF_Net_rx50_IRRG_DSM_v2', classes=7,
               m_views=3, kernel=8, hidden_ch=32,
               heads=1, depth=1,
               dropout=0.3, activation=None, shortcut=True, ndvi=False,
               ):
        # best mode for paper III so far
    if name == 'HPAF_Net_rx50_IRRG_DSM_v2':
        net = HPAF_Net_rx50_IRRG_DSM_v2(out_channels=classes, kernel=kernel, hidden_ch=hidden_ch,
                            heads=heads, depth=depth, m_views=m_views,
                            dropout=dropout, activation=activation,ndvi=ndvi,shortcut=shortcut,
                            )
    elif name == 'HPAF_Net_rx50_IRRG_DSM_v3':
        net = HPAF_Net_rx50_IRRG_DSM_v3(out_channels=classes, kernel=kernel, hidden_ch=hidden_ch,
                            heads=heads, depth=depth, m_views=m_views,
                            dropout=dropout, activation=activation,ndvi=ndvi,shortcut=shortcut,
                            )

    else:
        print('not found the net')
        return -1

    return net

# the best old model for Vaihigen, not using 1 - gate
# dense gate, directly mulitply by the Q vector.


class HPAF_Net_rx50_IRRG_DSM_v2(nn.Module):
    def __init__(self, in_ch = None, hidden_ch = 32, out_channels=6, kernel=16,
                 heads=1, depth=1, m_views=3,
                 dropout=0.2, activation=None,shortcut=True, ndvi=False,
                 ):
        super(HPAF_Net_rx50_IRRG_DSM_v2, self).__init__()

        self.in_ch_main = {'q': 256, 'z': 512, 'k': 1024}
        if in_ch is None:
            # self.in_ch = {'q': 80, 'z': 112, 'k': 160}
            self.in_ch = {'q': 40, 'z': 112, 'k': 960}
        else:
            self.in_ch = in_ch

        self.ndvi = ndvi

        if activation is None:
            self.activation = nn.PReLU()
        else:
            self.activation = activation
        self.num_cluster = out_channels

        modal_1 = se_resnext50_32x4d()
        self.m1_l0, self.m1_l1, self.m1_l2, self.m1_l3 = \
            modal_1.layer0, modal_1.layer1, modal_1.layer2, modal_1.layer3


        modal_2 = mobilenetv3_large()
        modal_2.load_state_dict(torch.load(mobilenet_path))
        # modal_2.features[0][0]

        self.m2_l1, self.m2_l2, self.m2_l3 = \
            modal_2.features[5:6], \
            modal_2.features[6:12], \
            torch.nn.Sequential(modal_2.features[12:16], modal_2.conv)  # h//32 w//32 960
        conv0 = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        for param in modal_2.features[0][0].parameters():
            par = param
            break

        conv0.parameters = par[:, [0], :, :]
        modal_2.features[0][0] = conv0
        self.m2_l0 = modal_2.features[0:5]  # output h//4 w//4 40

        self.paf_1 = PAF_block(in_ch=self.in_ch_main, hidden_ch=hidden_ch, kernel=kernel,
                               heads=heads, m_views=m_views,
                               activation=self.activation, shortcut=shortcut,
                               depth=depth, dropout=dropout)

        self.paf_2 = PAF_block(in_ch=self.in_ch, hidden_ch=hidden_ch, kernel=kernel,
                            heads=heads, m_views=m_views,
                            activation=self.activation, shortcut=shortcut,
                            depth=depth, dropout=dropout)


        self.gate = nn.Sequential(
            nn.Conv2d(hidden_ch,
                      self.in_ch['q'],
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_ch['q']),
            nn.MaxPool2d(2),
            # nn.Sigmoid(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(hidden_ch*2, out_channels,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
            # nn.MaxPool2d(2)
        )

        weight_xavier_init(self.final, self.paf_1, self.paf_2, self.gate)  # , self.att)

    def forward(self, x):
        x_size = x.size()
        qzk_set = dict()

        qzk_set['q'] = self.m1_l1(self.m1_l0(x[:, 0:3, :, :]))
        qzk_set['z'] = self.m1_l2(qzk_set['q']) # torch.cat((x2,y2), 1)
        qzk_set['k'] = self.m1_l3(qzk_set['z']) # torch.cat((x3,y3), 1)

        x1, A, loss = self.paf_1(qzk_set)
        g = self.gate(x1)

        qzk_set['q'] = self.m2_l1(self.m2_l0(x[:, [3], :, :])) * (1. - F.sigmoid(g)) + g
        qzk_set['z'] = self.m2_l2(qzk_set['q'])  # torch.cat((x2,y2), 1)
        qzk_set['k'] = self.m2_l3(qzk_set['z'])  # torch.cat((x3,y3), 1)

        x2, A, loss = self.paf_2(qzk_set)

        x1_size = x1.size()
        x2 = F.interpolate(x2, x1_size[2:], mode='bilinear', align_corners=False)

        x = self.final(torch.cat((x1,x2),1))

        if self.training:
            return F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False), loss
        else:
            return F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)



class HPAF_Net_rx50_IRRG_DSM_v3(nn.Module):
    def __init__(self, in_ch = None, hidden_ch = 32, out_channels=6, kernel=16,
                 heads=1, depth=1, m_views=3,
                 dropout=0.2, activation=None,shortcut=True, ndvi=False,
                 ):
        super(HPAF_Net_rx50_IRRG_DSM_v3, self).__init__()

        self.in_ch_main = {'q': 256, 'z': 512, 'k': 1024}
        if in_ch is None:
            # self.in_ch = {'q': 80, 'z': 112, 'k': 160}
            self.in_ch = {'q': 40, 'z': 112, 'k': 960}
        else:
            self.in_ch = in_ch

        self.ndvi = ndvi

        if activation is None:
            self.activation = nn.PReLU()
        else:
            self.activation = activation
        self.num_cluster = out_channels

        modal_1 = se_resnext50_32x4d()
        self.m1_l0, self.m1_l1, self.m1_l2, self.m1_l3 = \
            modal_1.layer0, modal_1.layer1, modal_1.layer2, modal_1.layer3


        modal_2 = mobilenetv3_large()
        modal_2.load_state_dict(torch.load(mobilenet_path))
        # modal_2.features[0][0]

        self.m2_l1, self.m2_l2, self.m2_l3 = \
            modal_2.features[5:6], \
            modal_2.features[6:12], \
            torch.nn.Sequential(modal_2.features[12:16], modal_2.conv)  # h//32 w//32 960
        conv0 = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        for param in modal_2.features[0][0].parameters():
            par = param
            break

        conv0.parameters = par[:, [0], :, :]
        modal_2.features[0][0] = conv0
        self.m2_l0 = modal_2.features[0:5]  # output h//4 w//4 40

        self.paf_1 = PAF_block(in_ch=self.in_ch_main, hidden_ch=hidden_ch, kernel=kernel,
                               heads=heads, m_views=m_views,
                               activation=self.activation, shortcut=shortcut,
                               depth=depth, dropout=dropout)

        self.paf_2 = PAF_block(in_ch=self.in_ch, hidden_ch=hidden_ch, kernel=kernel,
                            heads=heads, m_views=m_views,
                            activation=self.activation, shortcut=shortcut,
                            depth=depth, dropout=dropout)


        self.gate = nn.Sequential(
            nn.Conv2d(hidden_ch,
                      self.in_ch['q'],
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_ch['q']),
            nn.MaxPool2d(2),
            # nn.Sigmoid(),
        )

        self.gate2 = nn.Sequential(
            nn.Conv2d(self.in_ch['q'],
                      self.in_ch['q'],
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_ch['q']),
        )

        self.final = nn.Sequential(
            nn.Conv2d(hidden_ch*2, out_channels,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
            # nn.MaxPool2d(2)
        )

        weight_xavier_init(self.final, self.paf_1, self.paf_2, self.gate, self.gate2)  # , self.att)

    def forward(self, x):
        x_size = x.size()
        qzk_set = dict()

        qzk_set['q'] = self.m1_l1(self.m1_l0(x[:, 0:3, :, :]))
        qzk_set['z'] = self.m1_l2(qzk_set['q']) # torch.cat((x2,y2), 1)
        qzk_set['k'] = self.m1_l3(qzk_set['z']) # torch.cat((x3,y3), 1)

        x1, A, loss = self.paf_1(qzk_set)
        g = self.gate(x1)
        s = torch.sigmoid(g)


        qzk_set['q'] = self.m2_l1(self.m2_l0(x[:, [3], :, :])) * s + (1-s) *self.gate2(g)

        qzk_set['z'] = self.m2_l2(qzk_set['q'])  # torch.cat((x2,y2), 1)
        qzk_set['k'] = self.m2_l3(qzk_set['z'])  # torch.cat((x3,y3), 1)

        x2, A, loss = self.paf_2(qzk_set)

        x1_size = x1.size()
        x2 = F.interpolate(x2, x1_size[2:], mode='bilinear', align_corners=False)

        x = self.final(torch.cat((x1,x2),1))

        if self.training:
            return F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False), loss
        else:
            return F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)



