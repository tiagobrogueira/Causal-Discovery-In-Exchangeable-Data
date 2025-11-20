import numpy as np

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("EMD")
    except ModuleNotFoundError:
        return None  # or set to None


def gaussian_kernel(X, Y, sigma):
    """Compute the Gaussian kernel matrix between X and Y."""
    sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * X @ Y.T
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

def emd(d,lambda_=0.1, norm="uniform"):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    if max_points() is not None:
        n = min(max_points(), len(x))
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        y = y[idx]
        
    if norm=="uniform":
        scaler=MinMaxScaler()
    else:
        scaler=StandardScaler()
    x=scaler.fit_transform(x)
    y=scaler.fit_transform(y)

    L = x.shape[0]

    ux = np.random.rand(L, x.shape[1])
    uy = np.random.rand(L, y.shape[1])

    Kx=gaussian_kernel(x, x, 1)
    Ky=gaussian_kernel(y, y, 1)
    Kxux=gaussian_kernel(x, ux, 1)
    Kyuy=gaussian_kernel(y, uy, 1)
    Kux=gaussian_kernel(ux, ux, 1)
    Kuy=gaussian_kernel(uy, uy, 1)

    Vx = np.linalg.inv(Kx + np.eye(L) * lambda_ * L)
    Vy = np.linalg.inv(Ky + np.eye(L) * lambda_ * L)
        
    # Cxy terms
    first = (1 / L**2) * np.trace(Kx @ Ky) \
            - (2 / L**2) * np.trace(Ky @ Vx @ Kxux @ Kxux.T) \
            + (1 / L**2) * np.trace(Ky @ Vx @ Kxux @ Kux @ Kxux.T @ Vx)

    second = (1 / L**2) * np.trace(Ky @ Vx @ Kxux @ Kux @ Kxux.T @ Vx) \
             - (2 / L**2) * np.trace(Kyuy.T @ Vx @ Kxux @ Kux) \
             + (1 / L**2) * np.trace(Kux @ Kuy)

    # Cyx terms
    third = (1 / L**2) * np.trace(Ky @ Kx) \
            - (2 / L**2) * np.trace(Kx @ Vy @ Kyuy @ Kyuy.T) \
            + (1 / L**2) * np.trace(Kx @ Vy @ Kyuy @ Kuy @ Kyuy.T @ Vy)

    fourth = (1 / L**2) * np.trace(Kx @ Vy @ Kyuy @ Kuy @ Kyuy.T @ Vy) \
             - (2 / L**2) * np.trace(Kxux.T @ Vy @ Kyuy @ Kuy) \
             + (1 / L**2) * np.trace(Kuy @ Kux)

    Cxy = first + second
    Cyx = third + fourth
    return Cyx - Cxy
