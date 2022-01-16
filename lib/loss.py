import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy2D(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-1):
        super(CrossEntropy2D, self).__init__()

        self.loss = nn.NLLLoss(weight, reduction, ignore_index)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class ACW_loss(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5, ignore_index=255):
        super(ACW_loss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        # pred = F.softmax(prediction, 1)
        pred = torch.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        # one = torch.ones_like(err)

        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr # + self.weight
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        mfb = torch.clamp(mfb, min=0.01, max=1.0)
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None


        x = torch.einsum("ij->i", x)
        return x

    def forward(self, prediction, target):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """

        loss_ce = F.cross_entropy(prediction, target, ignore_index=self.ignore_index)
        pred = torch.softmax(prediction, 1)
        # pred = torch.sigmoid(prediction)
        # pred = torch.sigmoid(torch.log_softmax(prediction, 1))

        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        # bin_labels, valid_mask = expand_onehot_labels(target, pred.shape, self.ignore_index)
        # gt_proportion = get_region_proportion(bin_labels, valid_mask)
        # pred_proportion = get_region_proportion(pred, valid_mask)
        # # loss_reg = (pred_proportion - gt_proportion).abs().mean()
        # loss_reg = self.kl_div(gt_proportion, pred_proportion).mean()

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)
        err = torch.pow((one_hot_label - pred), 2)

        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)

        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union
        l1 = loss_pnc.mean()
        l2 = -dice.mean().log()

        # if self.itr%300:
        #     print(l1, loss_reg, loss_ce, l2)
        return l1 * loss_ce + l2 #+ loss_reg


    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr # + self.weight
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None
