import numpy as np


def eval_hist(pred, label, n):


    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iou(hist):


    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def eval_hist_crop(pred, target, unique_label):


    hist = eval_hist(pred.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist