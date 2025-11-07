#!/usr/bin/Rscript

NOFC=14         ### Maximum number of basis functions

##### Helper functions

### get domain size
domain = function(x){
    return(length(unique(sort(x))))
}
### na to zero
naTo0 = function(c){
    c[is.na(c)] = 0
    return(c)
}

##### Preprocessing

normalize = function(X){
    minX = min(X) - 0.000001
    maxX = max(X) + 0.000001
    X = (X - minX) / (maxX - minX)
    return(X)
}
### Normalization between zero and n
normX = function(x, n){
    if(min(x) == max(x)){
        return( rep(n, length(x)) )
    }else{
        return( ((x-min(x)) / (max(x) - min(x))) * n )
    }
}

getDensity = function(X, dens){
    xd = X
    indX = 1
    for(i in 1:length(X)){
        while(dens$x[indX] < X[i]){
            indX = indX + 1
        }
        if(indX > 1){
            xd[i] = (dens$y[indX-1] + dens$y[indX]) / 2
        }else{
            xd[i] = dens$y[indX]
        }
    }
    return(xd)
}

preprocessData = function(t,threshold,norm=T,standard=F){
    X = t[,1]
    Y = t[,2]
    N = length(X)
    ninit = N
    if(norm){
        if(!standard){
            X = normalize(X)
            Y = normalize(Y)
        }else{
            X = (X - mean(X)) / sd(X)
            Y = (Y - mean(Y)) / sd(Y)
        }
    }
    N_old = -1
    rounds = 1
    while(N_old != N){
        df = removeIsolatedPoints(X=X,Y=Y,threshold=threshold)
        if(dim(df)[1] == 0){
            break
        }
        X = df[,1]
        Y = df[,2]
        df = removeIsolatedPoints(X=Y,Y=X,threshold=threshold)
        if(dim(df)[1] == 0){
            break
        }
        Y = df[,1]
        X = df[,2]
        if(norm){
            if(!standard){
                X = normalize(X)
                Y = normalize(Y)
            }else{
                X = (X - mean(X)) / sd(X)
                Y = (Y - mean(Y)) / sd(Y)
            }
        }
        N_old = N
        N = length(X)
        if(rounds > 2){
            break
        }
        rounds = rounds + 1
    }
    print(paste(c(ninit, " -> ", N), collapse=""))
    return(data.frame(X,Y))
}

removeIsolatedPoints = function(X, Y, threshold){
    df = data.frame(X,Y)
    df = df[with(df, order(X)),]
    ## get median of pairwise dist
    di = dist(X)
    widthMed = median(di)
    BW = sqrt(2) * widthMed
    if(is.na(BW)){
        return(df)
    }else{
        d = density(X, bw=BW, kernel="gaussian")
        xd = getDensity(df$X, d)
        keep = (xd > threshold)
        df = df[keep,]
        return(df)
    }
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

##### Scoring functions

BICscore = function(rss, k, n){
    res = n * log2(rss/n + 1) + k * log2(n)
    return(res)
}
AICscore = function(rss, k, n){
    res = n * log2(rss/n + 1) + 2 * k
    return(res)
}
L0score = function(rss, k, n){
    res = rss + k * (n / 10000)
    return(res)
}
L0soft = function(rss, k, n){
    res = rss + k * (n / 100000)
    return(res)
}
RSSonlyscore = function(rss, k, n){
    return(rss)
}
error = function(res,x,y){
    yh = predict(res,x)$y
    eA = sum((y - yh)^2)
    return(eA)
}

##### Fitting algorithms

paramsInit = function(){
    return(list(a=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, ex=0, xm1=0, xm2=0,xm3=0, x8=0, x9=0, lx=0))
}
fitGeneric = function(y,X,pos){
    m = lm(y~1)
    X = data.frame(X)
    if(dim(X)[2] > 0){
        m = lm(y~.,X)
    }
    params = paramsInit()
    sse = tail(anova(m)[,2],1)
    coeff = naTo0(m$coefficients)
    pos = c(1,pos)
    for(i in 1:length(pos)){
        params[[pos[i]]] = coeff[i]
    }
    res = list(rss=sse, par=params, residuals=m$residuals)
    return(res)
}
forwardSelection = function(y,x,nof=NOFC,scoreF=BICscore){
    require("HapEstXXR") ## needed for powerset
    maxy = 1e+200
    miny = -1e+200
    lx = length(x)
    df = data.frame(x=x, x2=x^2, x3=x^3, x4=x^4, x5=x^5, x6=x^6, x7=x^7, expx=exp(x), xm1=x^(-1), xm2=x^(-2), xm3=x^(-3), x8=x^8, x9=x^9)
    if(min(x) < 0){
        nof = min(nof, NOFC-1) ## no log applicable
    }else{
        df = cbind(df, log=log(x))
    }
    # clean up
    for(i in 1:dim(df)[2]){
        c = df[,i]
        df[is.infinite(c) & c > 0, i] = 2147483647
        df[is.infinite(c) & c < 0, i] = -2147483647
        df[is.nan(c), i] = 0
    }
    # base rss for mean
    res = fitGeneric(y,data.frame(),c())
    score = scoreF(res$rss, 1, lx)
    left = 1:nof
    all = 1:nof
    bigX = c()
    rssv = c(res$rss)
    for(num in 1:nof){
        change = F ## did we improve this round?
        bigXR = c(bigX) ## needed for round best
        remaining = all[!(all %in% bigX)]
        for(i in remaining){
            currX = sort(c(bigX, i))
            currPos = currX + 1
            currRes = fitGeneric(y,df[,currX],currPos)
            if(currRes$rss == -1){
                next
            }
            currScore = scoreF(currRes$rss, length(currX)+1, lx)
            if(currScore < score){
                score = currScore
                res = currRes
                bigXR = currX
                change = T
            }
        }
        if(change){
            bigX = c(bigXR)
            rssv = c(rssv, res$rss)
        }else{
            break
        }
    }
    lrssv = length(rssv)
    if(lrssv < (NOFC+1)){
        rssv = c(rssv, rep(rssv[lrssv], (NOFC+1)-lrssv))
    }
    return(list(Score=score, RSS=res$rss, Params=res$par, Rssv=rssv, nParams=length(bigX)))
}
forwardBackwardSelection = function(y,x,nof=NOFC,scoreF=BICscore){
    require("HapEstXXR") ## needed for powerset
    maxy = 1e+200
    miny = -1e+200
    lx = length(x)
    df = data.frame(x=x, x2=x^2, x3=x^3, x4=x^4, x5=x^5, x6=x^6, x7=x^7, expx=exp(x), xm1=x^(-1), xm2=x^(-2), xm3=x^(-3), x8=x^8, x9=x^9)
    if(min(x) < 0){
        nof = min(nof, NOFC-1) ## no log applicable
    }else{
        df = cbind(df, log=log(x))
    }
    # clean up
    for(i in 1:dim(df)[2]){
        c = df[,i]
        df[is.infinite(c) & c > 0, i] = 2147483647
        df[is.infinite(c) & c < 0, i] = -2147483647
        df[is.nan(c), i] = 0
    }
    # base rss for mean
    res = fitGeneric(y,data.frame(),c())
    score = scoreF(res$rss, 1, lx)
    left = 1:nof
    bigX = c()
    for(num in 1:nof){
        change = F ## did we improve this round?
        bigXR = c(bigX) ## needed for round best
        for(i in 1:nof){
            currX = sort(c(bigX, i))
            currPos = currX + 1
            currRes = fitGeneric(y,df[,currX],currPos)
            if(currRes$rss == -1){
                next
            }
            currScore0 = L0soft(currRes$rss, length(currX)+1, lx)
            currScore = scoreF(currRes$rss, length(currX)+1, lx)
            currScore = min(currScore, currScore0)
            if(currScore < score){
                score = currScore
                res = currRes
                bigXR = currX
                change = T
            }
        }
        if(change){
            bigX = c(bigXR)
        }else{
            break
        }
    }
    ## is subset better?
    if(length(bigX) > 1){
        sink("/dev/null")
        S = powerset(bigX)
        sink()
        for(set in S){
            currX = unlist(set)
            currPos = currX + 1
            currRes = fitGeneric(y,df[,currX],currPos)
            if(currRes$rss == -1){
                next
            }
            currScore = scoreF(currRes$rss, length(currX)+1, lx)
            if(currScore < score){
                score = currScore
                res = currRes
                bigX = currX
            }
        }
    }
    ## return result
    return(list(Score=score, RSS=res$rss, Params=res$par, residuals=res$residuals, nParams=length(bigX)))
}
getSplineStatistic = function(y,x){
    # basic result
    yh = mean(y)
    rss0 = sum((y-yh)^2)
    # degree 2 result
    minDF = 2
    res = smooth.spline(x=x,y=y,df=minDF)
    # get levels
    maxDF = length(res$lev)
    rss = error(res,x,y)
    df = res$df
    # define result vectors
    dfs = c(1,df)
    rsss = c(rss0,rss)
    lastDF = df
    if(maxDF > minDF){
        for(i in (minDF+1):maxDF){
            resC = smooth.spline(x=x,y=y,df=i)
            rssC = error(resC,x,y)
            # stopping
            if(resC$df <= i & resC$df == lastDF){
                break
            }else{
                lastDF = resC$df
                rsss = c(rsss, rssC)
                dfs = c(dfs, lastDF)
            }
        }
    }
    rsss = rsss
    df = data.frame(DF=dfs, RSS=rsss)
    df = df[ with(df, order(DF)),]
    return(df)
}
fitSpline = function(y,x,nof=NOFC,scoreF=L0score, df=NULL){
    if(is.null(df)){
        df = getSplineStatistic(y,x)
    }
    nx = length(x)
    rss0 = df$RSS[1]
    score0 = scoreF(rss0, df$DF[1], nx)
    score = score0
    rss = rss0
    dims = df$DF[1]
    if(dim(df)[1] > 1){
        for(i in 2:dim(df)[1]){
            curr_score = scoreF(df$RSS[i], df$DF[i], nx)
            if(curr_score < score){
                score = curr_score
                rss = df$RSS[i]
                dims = df$DF[i]
            }
        }
    }
    return(list(Score=score, S0=score0, RSS=rss, R0=rss0, nParams=dims))
}
