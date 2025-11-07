#!/usr/bin/Rscript

#source("utilities.R")

##### Infer the best model

# number of function classes
NOFC = 8

##Params:
# a + bx + cx^2 + dx^3 + e*exp(x) + fx^-1 + gx^-2 + hx^-3 + ilog(x)
fofx = function(x, p){
    x.neg = x
    x.neg[x == 0] = resolution
    y.head = p$a + p$b * x + p$c * (x^2) + p$d * (x^3) + p$e * (exp(x)) + p$f * (x.neg^(-1)) + p$g * (x.neg^(-2)) + p$h * (x.neg^(-3)) + p$i*log(x.neg)
    #if(p$per != 0){
    #    y.head = y.head + p$si * sin(2*pi/p$per*x) + p$co * cos(2*pi/p$per*x)
    #}
    return(y.head)
}
fofX = function(X, coeff){
    
    return(y.head)
}

paramsInit = function(){
    return(list(a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0, per=0, si=0, co=0))
}
generateBinVector = function(k, l){
    bv = rep(0,l)
    i = l
    while(k > 0){
        r = 2^(i-1)
        if(k >= r){
            bv[i] = 1
            k = k - r
        }
        i = i - 1
    }
    return(bv)
}
getFunctionIndex = function(p){
    if(abs(p$b) > 0){
        return(1)
    }else if(abs(p$c) > 0){
        return(2)
    }else if(abs(p$d) > 0){
        return(3)
    }else if(abs(p$e) > 0){
        return(4)
    }else if(abs(p$f) > 0){
        return(5)
    }else if(abs(p$g) > 0){
        return(6)
    }else if(abs(p$h) > 0){
        return(7)
    }else if(abs(p$i) > 0){
        return(8)
    }else{
        return(0)
    }
}
###### Best fit
constantFit = function(y,x){
    m = lm(y~1)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitLin = function(y,x){

    m = lm(y~x)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$b = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitQuad = function(y,x){
    m = lm(y~I(x^2))
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$c = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitCub = function(y,x){
    m = lm(y~I(x^3))
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$d = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitExp = function(y,x){
    xe = exp(x)
    if(sum(is.infinite(xe)) > 0 | sum(is.nan(xe)) > 0){
        return(list(sse=-1))
    }
    m = lm(y~xe)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$e = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitNegLin = function(y,x){
    x[x == 0] = resolution
    xe = x^(-1)
    if(sum(is.infinite(xe)) > 0 | sum(is.nan(xe)) > 0){
        return(list(sse=-1))
    }
    m = lm(y~xe)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$f = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitNegQuad = function(y,x){
    x[x == 0] = resolution
    xe = x^(-2)
    if(sum(is.infinite(xe)) > 0 | sum(is.nan(xe)) > 0){
        return(list(sse=-1))
    }
    m = lm(y~xe)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$g = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitNegCub = function(y,x){
    x[x == 0] = resolution
    xe = x^(-3)
    if(sum(is.infinite(xe)) > 0 | sum(is.nan(xe)) > 0){
        return(list(sse=-1))
    }
    m = lm(y~xe)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$h = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitLog = function(y,x){
    x[x == 0] = resolution
    xe = log(x)
    if(sum(is.infinite(xe)) > 0 | sum(is.nan(xe)) > 0){
        return(list(sse=-1))
    }
    m = lm(y~xe)
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    params$a = coeff[1]
    params$i = coeff[2]
    yhead = y - fofx(x, params)
    rsq=log(1/sqrt(1-summary(m)$r.squared))
    res = list(sse=sse, model=parameterScore(m), par=params, residuals=yhead, gain=rsq)
    return(res)
}
fitGeneric = function(y,X,pos){
    keep = rep(T,length(pos))
    for(i in 1:length(pos)){
        if(sum(is.infinite(X[,i])) > 0 | sum(is.nan(X[,i])) > 0){
            keep[i] = F
        }
    }
    m = lm(y~1)
    pos = pos[keep]
    X = data.frame(X[,keep])
    if(sum(keep) > 0){
        m = lm(y~.,X)
    }
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    pos = c(1,pos)
    for(i in 1:length(pos)){
        params[[pos[i]]] = coeff[i]
    }
    res = list(sse=sse, model=parameterScore(m), par=params)
    return(res)
}
fitI = function(y,x,i){
    if(i > NOFC | i < 1){
        return(NULL)
    }else{
        if(i == 0){
            return(constantFit(y,x))
        }else if(i == 1){
            return(fitLin(y,x))
        }else if(i == 2){
            return(fitQuad(y,x))
        }else if(i == 3){
            return(fitCub(y,x))
        }else if(i == 4){
            return(fitExp(y,x))
        }else if(i == 5){
            return(fitNegLin(y,x))
        }else if(i == 6){
            return(fitNegQuad(y,x))
        }else if(i == 7){
            return(fitNegCub(y,x))
        }else{
            return(fitLog(y,x))
        }
    }
}
findBestFit = function(y,x,nof=NOFC){
    rr = fitI(y,x,1)
    s = gaussian_score_emp_sse(rr$sse, length(x)) + rr$model
    for(i in 2:nof){
        rr2 = fitI(y,x,i)
        if(rr2$sse == -1){
            next
        }
        s2 = gaussian_score_emp_sse(rr2$sse, length(x)) + rr2$model
        if(s2 < s){
            s = s2
            rr = rr2
        }
    }
    return(rr)
}
findBestMixedFit = function(y,x,nof=NOFC){
    x2 = x
    x2[x2 == 0] = resolution
    df = data.frame(x=x, x2=x^2, x3=x^3, expx=exp(x2), xm1=x2^(-1), xm2=x2^(-2), xm3=x2^(-3), logx=log(x2))
    params = c(2:(nof+1))
    res_o = fitI(y,x,1)
    score = gaussian_score_emp_sse(res_o$sse, length(x)) + res_o$model
    for(i in 2:((2^nof)-1)){
        setup = generateBinVector(i, nof)
        currX = data.frame(df[,setup == 1])
        currPos = params[setup == 1]
        res = fitGeneric(y,currX,currPos)
        if(res$sse == -1){
            next
        }
        curr_score = gaussian_score_emp_sse(res$sse, length(x)) + res$model
        if(curr_score < score){
            score = curr_score
            res_o = res
        }
    }
    return(res_o)
}

fitComparison = function(fit, l, lx, score, bins_old, nof=NOFC){
    newScore = gaussian_score_emp_sse(fit$sse, l) + fit$model
    if(bins_old <= 1){
        newScore = newScore + logg(nof)
    }
    delta = newScore - score + log2nChoosek(lx-1, bins_old+1) + logN(bins_old+1) - logN(bins_old) - log2nChoosek(lx-1, bins_old)
    return(delta)
}

fitWrapper = function(y,x,nof=NOFC,bestFit=findBestFit){
    minNum = 50000
    maxX = max(x)
    xf = round(x*100000)
    tx = table(xf)
    if(length(tx) <= 2){  ## binary data
        return(list(s=0, b=-2, p=-2))
    }
    mx = mean(tx)
    sd = sd(tx)
    lx = length(x)
    res = bestFit(y,x,nof=nof)
    fun = getFunctionIndex(res$par)
    score = gaussian_score_emp_sse(res$sse, length(x))
    modelCosts = logN(1) + res$model
    costs = score + modelCosts
    xi = xf %in% as.numeric(names(tx)[tx >= minNum])
    xfg = xf[xi]
    xg = x[xi]
    yg = y[xi]
    scores = rep(score, nof)
    bins = rep(1,nof)
    for(e in unique(sort(xfg))){
        ones = xfg == e
        yt = sort(yg[ones])
        xt = normX(1:length(yt), 10) - 5.0
        curr_l = length(xt)
        old_sse = sum((yg[ones] - fofx(xg[ones], res$par))^2)
        old_score = gaussian_score_emp_sse(old_sse, curr_l)
        for(i in 1:(nof)){
            fit = fitI(yt, xt, i)
            if(fit$sse == -1){
                next
            }
            sFit = fitComparison(fit, curr_l, lx, old_score, bins[i],nof=nof)
            if(sFit < 0){
                bins[i] = bins[i] + 1
                scores[i] = scores[i] + sFit
            }
        }
    }
    # correct for all divided
    for(j in 1:length(bins)){
        if(bins[j] > length(tx)){
            bins[j] = bins[j] - 1
        }
    }
    i = which(scores == min(scores))[1]
    score = scores[i]
    costs = score
    potential_bins = sum(tx >= minNum)
    if(potential_bins != length(tx)){
        potential_bins = potential_bins + 1
    }
    if(bins[i] > 1){
        fun = i
    }
    return(list(s=score, b=bins[i], p=potential_bins, f=fun))
}

defaultScore = function(x, v=1){
    score = 0 #min(gaussian_score(x), -logg(resolution) * length(x))
    if(v == 2){
        score = gaussian_score(x)
    }else{
        score = -logg(resolution) * length(x)
    }
    return(score)
}

### Slope algorithm
# @t two dim data frame containing x and y
# @mixedFunctions cosider all possible combinations of functions, i.e. f(y) = ax + b*log(x) + c*x^-3
# @nof nof=6 refers to the version of the conference paper; nof 8 to the journal version
# @alpha alpha cut off for sginificance test based on absolute difference
Slope = function(t, mixedFunctions=F, nof=6, alpha=0.001){
    if(nof > NOFC){
        nof = NOFC
    }
    x = normX(t[,1],1)
    y = normX(t[,2],1)

    setResolution(mindiff(y))
    dy = defaultScore(y)
    resXtoY = 0
    if(mixedFunctions){
        resXtoY = fitWrapper(y,x,nof=nof,bestFit=findBestMixedFit)
    }else{
        resXtoY = fitWrapper(y,x,nof=nof,bestFit=findBestFit)
    }
    sseXtoY = resXtoY$s
    
    setResolution(mindiff(x))
    dx = defaultScore(x)
    resYtoX = 0
    if(mixedFunctions){
        resYtoX = fitWrapper(x,y,nof=nof,bestFit=findBestMixedFit)
    }else{
        resYtoX = fitWrapper(x,y,nof=nof,bestFit=findBestFit)
    }
    sseYtoX = resYtoX$s
    dXY = sseXtoY + dx
    dYX = sseYtoX + dy
    dXtoY = dXY / (dx + dy)
    dYtoX = dYX / (dx + dy)
    
    # Get delta
    eps = dXtoY - dYtoX
    pv = 2^(-(abs(dXY - dYX)/2))
    ## if data is binary
    if(resXtoY$p == -2 | resYtoX$p == -2){
        eps = 0.0
        pv = 1.0
    }
    
    # Determine causal direction
    causd = "--"
    compression = 0
    if(abs(eps) > 0.0 & pv < alpha){
        if(eps < 0){
            causd = "->"
            compression = (dy - sseXtoY) / dy
        }else{
            causd = "<-"
            compression = (dx - sseYtoX) / dx
        }
    }
    r = list(epsilon = eps, cd = causd, p.value=pv, sc=c(dXY, dYX))
    return(r)
}
