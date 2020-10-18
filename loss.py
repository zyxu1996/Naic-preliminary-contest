import torch
import torch.nn as nn


def one_hot(index_tensor, cls_num):
    b, h, w = index_tensor.size()
    index_tensor = index_tensor.view(b, 1, h, w)
    one_hot_tensor = torch.cuda.FloatTensor(b, cls_num, h, w).zero_()
    one_hot_tensor = one_hot_tensor.cuda(index_tensor.get_device())
    target = one_hot_tensor.scatter_(1, index_tensor.long(), 1)

    return target


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing=0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            target = one_hot(target, 8)
            logprobs = self.log_softmax(x)
            nll_loss = -logprobs * target

            nll_loss = nll_loss.sum(1)
            smooth_loss = -logprobs.mean(1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return nn.CrossEntropyLoss(x, target)


class edge_weak_loss(nn.Module):
    def __init__(self):
        super(edge_weak_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, scale_pred, target, edge):
        edge_loss = (torch.mul(self.ce_loss(scale_pred, target), torch.where(
            edge == 0, torch.tensor([1.]).cuda(), torch.tensor([0.5]).cuda()))).mean()

        return edge_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)

        return loss