#!/usr/bin/Rscript

#source("utilities.R")

##### Sloppy

### A list of parameters can be passed. E.g. params = list(alpha=0.01, stand=F, threshold=0.05, nof=9), where
### @alpha: can be used to set a threshold on the confidence value C; i.e. to decide for a direction, C >= alpha must hold
### @stand: boolean value; if true the data is standardized, if false it is normalized
### @threshold: cut-off to filter out the most outlying points (see utilities.R)
### @nof: (ONLY IF BASIS FUNCTIONS ARE USED; FOR SPLINES NOT CONSIDERD) is the number of basis functions that should be considered: the basis functions are oredered as follows; f(x) = : 1) x, 2) x^2, ... 7) x^7, 8) e^x 9) x^-1, ... 11) x^-3, 12) x^8, 13) x^9, 14) log x (see forwardBackwardSelection in utilities.R)

SloppyS = function(t, scoreF=BICscore, params=NULL){
    return(Sloppy(t, scoreF=scoreF, params=params, fun=fitSpline))
}
SloppyP = function(t, scoreF=BICscore, params=NULL){
    return(Sloppy(t, scoreF=scoreF, params=params, fun=forwardBackwardSelection))
}
Sloppy = function(t, fun=fitSpline, scoreF=BICscore, params=NULL){
    ## read & check params
    stand=T
    nof=9
    alpha=0.0
    threshold=0.0
    if(is.null(params)){
        params = list(nof=9, threshold=0.0, alpha=0.0, stand=F)
    }
    stand = params$stand
    alpha = params$alpha
    nof = params$nof
    threshold = params$threshold
    if(nof > NOFC){
        nof = NOFC
    }
    
    ## preprocess data
    x = t[,1]
    y = t[,2]
    if(domain(x) <= 4 | domain(y) <= 4){
        return(list(epsilon = 0.0, cd = "--", XtoY.meta=0.0, YtoX.meta=0.0))
    }
    if(threshold == 0.0){
        if(stand){
            x = (x - mean(x)) / sd(x)
            y = (y - mean(y)) / sd(y)
        }else{
            x = normX(t[,1],1)
            y = normX(t[,2],1)
        }
    }else{
        pt = preprocessData(t,threshold,stand=stand)
        x = pt[,1]
        y = pt[,2]
    }
    ## calculate causal direction
    resXtoY = fun(y,x,scoreF=scoreF,nof=nof)
    resYtoX = fun(x,y,scoreF=scoreF,nof=nof)
    
    # Determine causal direction
    causd = "--"
    errorRatio = min(resXtoY$Score, resYtoX$Score) / max(resXtoY$Score, resYtoX$Score)
    eps = 0
    if(errorRatio <= 1.0 - alpha){
        eps = resXtoY$Score - resYtoX$Score
    }
    nParams = 0
    if(abs(eps) > 0.0){
        if(eps < 0){
            causd = "->"
            eps = -(1-errorRatio)
            nParams = resXtoY$nParams
        }else{
            causd = "<-"
            eps = 1 - errorRatio
            nParams = resYtoX$nParams
        }
    }
    print(nParams)
    r = list(epsilon = eps, cd = causd, XtoY.meta=resXtoY, YtoX.meta=resYtoX, nParams=nParams, ER=errorRatio)
    return(r)
}
