import numpy as np
import math
from scipy.optimize import curve_fit
from sklearn.preprocessing import minmax_scale
np.seterr(divide='ignore')

def resolution(x):
    x = np.sort(np.asarray(x))
    diffs = np.diff(x)
    res = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 0
    if res<=0:
        return np.nan
    return res

def H(X, eps):
    X = np.asarray(X)
    n_bins = int(1 / eps)

    # Create equally spaced bins between min and max of X
    bins = np.linspace(X.min(), X.max(), n_bins + 1)
    counts, _ = np.histogram(X, bins=bins)

    # Normalize counts to probabilities
    N = len(X)
    probs = counts[counts > 0] / N  # ignore empty bins

    # Shannon entropy
    H = -np.sum(probs * np.log(probs))
    return H

def information_dimension(x,epsmin=-5, epsmax=-0.5, num_eps=100, plots=False):
    epsilons = np.logspace(epsmin, epsmax, num_eps) #epsmin and epsmax are in log10 scale.
    Hs=[H(x,ei) for ei in epsilons]
    coeffs = np.polyfit(np.log(epsilons), Hs, 1)
    lx=-coeffs[0]
    return lx

def mdl_lengthv2(resid, k):
    """Returns L(D|M)+L(M) in bits."""
    N=len(resid)
    L_normal = (N/2)*np.log2(2*math.pi*np.mean(resid**2)) + N/(2*np.log(2))
    L_model = (k/2)*np.log2(N)
    L_data=L_normal
    #print("lengths:", L_data, L_model)
    return L_data + L_model

def fit_polynomial(x, y, max_deg=5):
    """MDL-select best polynomial up to degree max_deg."""
    N = x.size
    best = (np.inf, None, None, None)  # (mdl, degree, coeffs, resid)
    for deg in range(1, max_deg+1):
        coeffs = np.polyfit(x, y, deg)
        y_pred = np.polyval(coeffs, x)
        resid = y - y_pred
        mdl = mdl_lengthv2(resid, k=deg+1)
        if mdl < best[0]:
            best = (mdl, deg, coeffs, resid)
    return best  # (mdl, degree, coeffs, resid)

def fit_reciprocal(x, y, max_pow=2):
    """MDL-select between a/x^p + b for p=1..max_pow."""
    N = x.size
    best = (np.inf, None, None, None)  # (mdl, power, [a,b], resid)
    for p in range(1, max_pow+1):
        def func(x, a, b): return a/(x**p) + b
        popt, _ = curve_fit(func, x, y)
        y_pred = func(x, *popt)
        resid = y - y_pred
        mdl = mdl_lengthv2(resid, k=2)
        if mdl < best[0]:
            best = (mdl, p, popt, resid)
    return best  # (mdl, power, params, resid)

def fit_explog(x, y):
    """MDL-select best of a*exp(x)+b vs a*log(x)+b."""
    N = x.size
    candidates = []
    # exponential
    popt_e, _ = curve_fit(lambda x,a,b: a*np.exp(x)+b, x, y)
    y_e = popt_e[0]*np.exp(x) + popt_e[1]
    resid_e = y - y_e
    candidates.append(("exp", popt_e, mdl_lengthv2(resid_e, 2), resid_e))
    # logarithm (domain check)
    if np.all(x > 0):
        popt_l, _ = curve_fit(lambda x,a,b: a*np.log(x)+b, x, y)
        y_l = popt_l[0]*np.log(x) + popt_l[1]
        resid_l = y - y_l
        candidates.append(("log", popt_l, mdl_lengthv2(resid_l, 2), resid_l))

    # pick best
    return min(candidates, key=lambda t: t[2])  # (label, popt, mdl, resid)

def compute_mdl_fit_v2(x, y):
    """
    Perform MDL-based model selection among polynomial, reciprocal, and exp/log families.

    Returns:
        model_desc: description string of the selected model
        params: tuple or array of fitted parameters
        residuals: array of y - y_pred for each sample
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)

    # apply those indices to both x and y
    x = x[idx]
    y = y[idx]

    N = x.size

    # 1) best polynomial
    mdl_p, deg_p, coeffs_p, resid_p = fit_polynomial(x, y, max_deg=5)

    # 2) best reciprocal
    mdl_r, pow_r, params_r, resid_r = fit_reciprocal(x, y, max_pow=2)

    # 3) best exp/log
    label_el, params_el, mdl_el, resid_el = fit_explog(x, y)

    # Compare the three families
    candidates = [
        ("poly", mdl_p, f"poly degree {deg_p}", coeffs_p, resid_p),
        ("reciprocal", mdl_r, f"a/x^{pow_r}+b", params_r, resid_r),
        ("exp/log", mdl_el, f"a*{label_el}(x)+b", params_el, resid_el),
    ]
    # select best
    fam, best_mdl, model_desc, params, residuals = min(candidates, key=lambda t: t[1])

    return residuals, best_mdl




def rdmdl(d, delta="nbased",scaling="minmax", estimator="rd",diff=False):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    
    if scaling == "minmax":
        x = minmax_scale(x)
        y = minmax_scale(y)
    elif scaling == "normalize":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    x=np.array(x).flatten()
    y=np.array(y).flatten()

    if delta=="nbased":
        dx=1/len(x)
        dy=dx
    elif delta=="res":
        dx=min(resolution(x), resolution(y))**2/12
        dy=dx
    elif delta=="sturges":
        dx=1/((1+np.log2(len(x)))**2)
        dy=dx
    elif delta=="rice":
        dx=1/((2*len(x)**(1/3))**2)
        dy=dx
    elif delta=="scott":
        stdx=np.std(x)
        stdy=np.std(y)
        hx=3.5*stdx/(len(x)**(1/3))
        hy=3.5*stdy/(len(y)**(1/3))
        if diff==False:
            dx= min(hx,hy)**2
            dy=dx
        else:
            dx=hx**2
            dy=hy**2
    elif delta=="freedman-diaconis":
        q75x, q25x = np.percentile(x, [75 ,25])
        q75y, q25y = np.percentile(y, [75 ,25])
        iqr_x = q75x - q25x
        iqr_y = q75y - q25y
        hx=2*iqr_x/(len(x)**(1/3))
        hy=2*iqr_y/(len(y)**(1/3))
        if diff==False:
            dx= min(hx,hy)**2
            dy=dx
        else:
            dx=hx**2
            dy=hy**2

    if estimator=="rd":
        lx=len(x)*information_dimension(x)*np.log2(1/dx)/2
        ly=len(y)*information_dimension(y)*np.log2(1/dy)/2
    elif estimator=="hist":
        lx=len(x)*H(x,1/np.sqrt(dx))
        ly=len(y)*H(y,1/np.sqrt(dy))
    _,lxy=compute_mdl_fit_v2(x,y)
    _,lyx=compute_mdl_fit_v2(y,x)
    lxtoy=lx+ lxy
    lytox=ly+ lyx

    guess=(lytox-lxtoy)/len(x) #indepedent of n 
    return guess

def rdmdl_lx(d,scaling="minmax",delta="nbased",estimator="rd",diff=False):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    
    if scaling == "minmax":
        x = minmax_scale(x)
        y = minmax_scale(y)
    elif scaling == "normalize":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    x=np.array(x).flatten()
    y=np.array(y).flatten()

    if delta=="nbased":
        dx=1/len(x)
        dy=dx
    elif delta=="res":
        dx=min(resolution(x), resolution(y))**2/12
        dy=dx
    elif delta=="sturges":
        dx=1/((1+np.log2(len(x)))**2)
        dy=dx
    elif delta=="rice":
        dx=1/((2*len(x)**(1/3))**2)
        dy=dx
    elif delta=="scott":
        stdx=np.std(x)
        stdy=np.std(y)
        hx=3.5*stdx/(len(x)**(1/3))
        hy=3.5*stdy/(len(y)**(1/3))
        if diff==False:
            dx= min(hx,hy)**2
            dy=dx
        else:
            dx=hx**2
            dy=hy**2
    elif delta=="freedman-diaconis":
        q75x, q25x = np.percentile(x, [75 ,25])
        q75y, q25y = np.percentile(y, [75 ,25])
        iqr_x = q75x - q25x
        iqr_y = q75y - q25y
        hx=2*iqr_x/(len(x)**(1/3))
        hy=2*iqr_y/(len(y)**(1/3))
        if diff==False:
            dx= min(hx,hy)**2
            dy=dx
        else:
            dx=hx**2
            dy=hy**2

    if estimator=="rd":
        lx=len(x)*information_dimension(x)*np.log2(1/dx)/2
        ly=len(y)*information_dimension(y)*np.log2(1/dy)/2
    elif estimator=="hist":
        lx=len(x)*H(x,1/np.sqrt(dx))
        ly=len(y)*H(y,1/np.sqrt(dy))
    return ly-lx

def rdmdl_lyx(d,scaling="minmax"):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    
    if scaling == "minmax":
        x = minmax_scale(x)
        y = minmax_scale(y)
    elif scaling == "normalize":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

    x=np.array(x).flatten()
    y=np.array(y).flatten()

    _,lxy=compute_mdl_fit_v2(x,y)
    _,lyx=compute_mdl_fit_v2(y,x)
    return lyx-lxy  