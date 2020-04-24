import numpy as np
from scipy.special import expit, xlogy
import Pyro4

from logistic_client import n_nodes, n_obs, n_cols


@Pyro4.expose
class LocalNode:
    def __init__(self):
        self.X = np.random.rand(n_obs, n_cols)
        self.y = np.random.randint(0, 2, n_obs)

    def get_local_parameters(self, coeff):
        coeff = np.array(coeff)
        X, y = self.X, self.y

        z = X @ coeff
        s = expit(z)
        d = s * (1 - s)
        D = np.diag(d)

        hess = X.T @ D @ X
        y_ratio = (y - s) / d
        y_ratio[(y == 0) & (s == 0)] = -1
        y_ratio[(y == 1) & (s == 1)] = 1

        grad = X.T @ D @ (z + y_ratio)

        ll = np.sum(xlogy(y, s) + xlogy(1 - y, 1 - s))

        return grad.tolist(), hess.tolist(), ll


def start_server():
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    [
        ns.register(f"local_node{i}", daemon.register(LocalNode()))
        for i in range(n_nodes)
    ]
    daemon.requestLoop()


if __name__ == "__main__":
    start_server()
