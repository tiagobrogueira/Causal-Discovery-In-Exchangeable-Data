from os import name
import numpy as np
from scipy import stats


def fit_model(X, Y, cutpoint):
    part1 = X <= cutpoint
    part2 = np.logical_not(part1)
    n1 = np.sum(part1)
    n2 = np.sum(part2)
    n = n1+n2
    X1 = X[part1]
    Y1 = Y[part1]
    X2 = X[part2]
    Y2 = Y[part2]
    if(len(np.unique(X1))<5 or len(np.unique(X2))<5 ):
        return False,0,0

    _, r1, _, _, _ = np.polyfit(X1, Y1, 1, full=True)
    _, r2, _, _, _ = np.polyfit(X2, Y2, 1, full=True)
    var1 = np.var(Y1)*n1
    var2 = np.var(Y2)*n2
    R1 = 1 - r1/var1
    R2 = 1 - r2/var2
    R_total = n1/n*R1+n2/n*R2
    return True, r1+r2, R_total


def test_invertible(X, Y, cutpoint_x, cutpoint_y, eta, sample_size=1000, significance=0.05):
    part1X = X <= cutpoint_x
    part2X = np.logical_not(part1X)
    nl = np.sum(part1X)
    nr = np.sum(part2X)
    n = nl+nr
    corr_left, _ = stats.pearsonr(X[part1X], Y[part1X])
    corr_right, _ = stats.pearsonr(X[part2X], Y[part2X])
    z_l = np.random.normal(np.arctan(corr_left), 1/(nl-3), sample_size)
    z_r = np.random.normal(np.arctan(corr_right), 1/(nr-3), sample_size)
    R0_XY = (nl*np.tanh(z_l)**2+nr*np.tanh(z_r)**2)/n

    part1Y = Y <= cutpoint_y
    part2Y = np.logical_not(part1Y)
    nl = np.sum(part1Y)
    nr = np.sum(part2Y)
    n = nl+nr
    corr_left, _ = stats.pearsonr(X[part1Y], Y[part1Y])
    corr_right, _ = stats.pearsonr(X[part2Y], Y[part2Y])
    z_l = np.random.normal(np.arctan(corr_left), 1/(nl-3), sample_size)
    z_r = np.random.normal(np.arctan(corr_right), 1/(nr-3), sample_size)
    R0_YX = (nl*np.tanh(z_l)**2+nr*np.tanh(z_r)**2)/n
    eta_0 = np.maximum(R0_YX/R0_XY, R0_XY/R0_YX)
    p_value = np.mean(eta > eta_0)
    return p_value >= 1-significance


def non_invertible_causal(X, Y):
    cutpoints = np.quantile(X, np.linspace(0, 1, 20))[1:-1]
    init = True
    best = 0
    R_XY = 0
    XY_cutpoint = 0
    for quantile in cutpoints:
        success, residual, R_squared = fit_model(X, Y, quantile)
        if not success:
            continue
        if init or residual < best:
            init = False
            best = residual
            R_XY = R_squared
            XY_cutpoint = quantile

    cutpoints = np.quantile(Y, np.linspace(0, 1, 20))[1:-1]
    init = True
    best = 0
    R_YX = 0
    YX_cutpoint = 0
    for quantile in cutpoints:
        success, residual, R_squared = fit_model(Y, X, quantile)
        if not success:
            continue
        if init or residual < best:
            init = False
            best = residual
            R_YX = R_squared
            YX_cutpoint = quantile
    eta = 0
    direction = 1
    if(R_XY > R_YX):
        eta = R_XY/R_YX
    else:
        eta = R_YX/R_XY
        direction = 0
    is_invertible = test_invertible(X, Y, XY_cutpoint, YX_cutpoint, eta)
    return direction, is_invertible, eta


if __name__ == "__main__":
    f = open("../pairs/pairmeta.txt")
    causal_direction_gt = np.zeros(108)
    weights = np.zeros(108)
    for i in range(108):
        line = f.readline()
        information = line.split(" ")[1]
        if information == "1":
            causal_direction_gt[i] = 1
        else:
            causal_direction_gt[i] = 0
        weights[i] = line.split()[5]
    f.close()

    causal_direction_pred = np.zeros(108)
    causal_direction_pred_r2 = np.zeros(108)
    skipped = []
    confident = []
    confidence_scores = np.zeros(108)
    show = False

    for i in range(1, 109):
        if show:
            pass
            i = np.random.randint(1, high=109)
        if i in [47, 54, 55, 70, 71, 105, 107]:
            skipped.append(i-1)
            continue
        datapath = "../pairs/pair{:04d}.txt".format(i)
        data = np.loadtxt(datapath)
        xsrc = data[:, 0]
        ysrc = data[:, 1]

        direction, is_invertible, eta = non_invertible_causal(xsrc, ysrc)

        if is_invertible:
            confident.append(i-1)

        print("_______________________________________________________")
        if causal_direction_gt[i-1]:
            truth = "X->Y"
        else:
            truth = "Y->X"

        if direction:
            causal_direction_pred[i-1] = 1
            pred = "X->Y"
        else:
            causal_direction_pred[i-1] = 0
            pred = "Y->X"

        confidence_scores[i-1] = eta

        if causal_direction_pred[i-1] == causal_direction_gt[i-1]:
            result = "correct"
        else:
            result = "false"
        print("Pair{:04d} {:s}: Truth {:s} | Predicted {:s}".format(
            i, result, truth, pred))

    confident_decisions = causal_direction_gt[confident] == causal_direction_pred[confident]
    print("Confident decisions:", len(confident_decisions),
          "| Accuracy:", np.mean(confident_decisions))

    causal_direction_gt = np.delete(causal_direction_gt, skipped)
    causal_direction_pred = np.delete(causal_direction_pred, skipped)
    weights = np.delete(weights, skipped)
    confidence_scores = np.delete(confidence_scores, skipped)

    correct = causal_direction_gt == causal_direction_pred
    weighted_accuracy = sum(correct * weights)/sum(weights)

    print(len(correct), "pairs decided")
    print("Overall Accuracy:", np.mean(correct),
          "| Weighted:", weighted_accuracy)
