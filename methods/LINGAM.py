# %%
import numpy as np
from sklearn.decomposition import FastICA

# %%
def permuteW(W):
    score1=1/W[0,0]+1/W[1,1]
    score2=1/W[0,1]+1/W[1,0]
    if score1<score2:
        return W
    else:
        backup=W[0]
        W[0]=W[1]
        W[1]=backup
        return W

def switchB(B):
    P = np.array([[0, 1], [1, 0]])
    return P @ B @ P.T

def lingam(d):
    x,y=d
    X=np.concatenate((x,y),axis=1)
    if X.shape[0]<3:
        return np.nan
    ica = FastICA()
    ica.fit_transform(X)
    W = ica.components_        # W=I-B
    Wp=permuteW(W)
    # Divide each row of Wp by its diagonal element (row scaling)
    diag_elements = np.diag(Wp)
    # Avoid division by zero (assumes nonzero diagonal)
    Wp = Wp / np.expand_dims(diag_elements, axis=1)

    B = np.eye(Wp.shape[0]) - Wp
    #find permutation of B
    return B[0,1]**2-B[1,0]**2


