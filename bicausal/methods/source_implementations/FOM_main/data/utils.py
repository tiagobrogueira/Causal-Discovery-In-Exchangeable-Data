import numpy as np


def sampling(x, y, n):
    if len(x) > n:
        if len(x) != len(y):
            raise (Exception, "lenght of x must same as y")
        ind = np.random.permutation(len(x))
        x = x[ind[:n]]
        y = y[ind[:n]]
        return x, y
    return x, y


def cal_decision_rate(scores, labels=None, weights=None, step=0.25):
    if labels == None and weights == None:
        labels = np.ones(len(scores))
        weights = np.ones(len(scores))

    s = np.abs(scores)
    weights = np.array(weights)
    w_ = np.zeros(len(s))
    ind = np.argsort(-s)

    steps = np.arange(step, 1 + step, step)
    end = len(s) * steps
    out = []
    for i in range(len(s)):
        if (scores[i] > 0. and labels[i] == 1) or (scores[i] < 0.
                                                   and labels[i] == 0):
            w_[i] = weights[i]
        else:
            w_[i] = 0
    # print(w_)
    for i in end.astype(int):
        out.append(np.sum(w_[ind[:i]]) / np.sum(weights[ind[:i]]))
    return steps.tolist(), out
