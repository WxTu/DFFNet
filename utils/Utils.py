import numpy as np
import torch.nn.functional as F


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)

        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        acc_cls = tp / (sum_a1 + np.finfo(np.float32).eps)
        acc_cls = np.nanmean(acc_cls)

        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall_Acc': acc,
                'Mean_Acc': acc_cls,
                'FreqW_Acc': fwavacc,
                'Mean_IoU': mean_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class OneCycle(object):
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt=10, div=10):
        self.nb = nb
        print(self.nb)
        self.div = div
        self.step_len = int(self.nb * (1 - prcnt / 100) / 2)
        print(self.step_len)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []

    def calc(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return lr, mom

    def calc_lr(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr / self.div)
            return self.high_lr / self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * (1 - 0.99 * ratio) / self.div
        elif self.iteration > self.step_len:
            ratio = 1 - (self.iteration - self.step_len) / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.iteration / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration - self.step_len) / self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.iteration / self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom


def poly_lr_scheduler(optimizer, init_lr, iter_num, lr_decay_iter=1, max_iter=89280, power=0.9):
    curr_lr = init_lr

    if iter_num % lr_decay_iter or iter_num > max_iter:
        return curr_lr

    for param_group in optimizer.param_groups:
        curr_lr = init_lr * (1 - iter_num / max_iter) ** power

        param_group['lr'] = curr_lr

    return curr_lr


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:

        if param_group['lr'] > 0.0:
            param_group['lr'] = lr

    return optimizer


def update_aggregated_weight_average(model, weight_aws, full_iter, cycle_length):
    for name, param in model.named_parameters():
        n_model = full_iter / cycle_length
        weight_aws[name] = (weight_aws[name] * n_model + param.data) / (n_model + 1)

    return weight_aws


def cross_entropy2d(input_data, target, weight=None, size_average=True):
    n, c, h, w = input_data.size()
    nt, ht, wt = target.size()

    if h > ht and w > wt:
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt:
        input_data = F.upsample(input_data, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    log_p = F.log_softmax(input_data)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss
