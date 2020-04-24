import numpy as np
import Pyro4

n_obs = 1000
n_nodes = 15
n_cols = 5
ntot_obs = n_obs * n_nodes

local_nodes = [Pyro4.Proxy(f"PYRONAME:local_node{i}") for i in range(n_nodes)]


def logistic_regression():
    coeff, ll = init_model(n_cols, ntot_obs)
    while True:
        print(ll)
        res = [ln.get_local_parameters(coeff.tolist()) for ln in local_nodes]
        grad = sum(np.array(r[0]) for r in res)
        hess = sum(np.array(r[1]) for r in res)
        ll_new = sum(r[2] for r in res)

        coeff = update_coefficients(grad, hess)
        if abs(ll - ll_new) <= 1e-6:
            break
        ll = ll_new
    print(coeff)


def init_model(n_cols, n_obs):
    ll = -2 * n_obs * np.log(2)
    coeff = np.zeros(n_cols)
    return coeff, ll


def update_coefficients(grad, hess):
    covariance = np.linalg.inv(hess)
    coeff = covariance @ grad
    return coeff


if __name__ == "__main__":
    import time

    s = time.perf_counter()
    logistic_regression()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
