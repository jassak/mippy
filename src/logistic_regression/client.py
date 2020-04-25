import numpy as np
import Pyro4

from logistic_regression import properties, n_nodes, n_obs, n_cols, ntot_obs


local_nodes = [Pyro4.Proxy(f"PYRONAME:local_node{i}") for i in range(n_nodes)]


def logistic_regression():
    coeff, ll = init_model(n_cols, ntot_obs)
    while True:
        print(ll)
        res = [ln.get_local_parameters(coeff.tolist()) for ln in local_nodes]
        grad, hess, ll_new = merge_local_results(res)

        coeff = update_coefficients(grad, hess)
        if abs(ll - ll_new) <= 1e-6:
            break
        ll = ll_new
    print(coeff)


def merge_local_results(res: list):
    grad = sum(np.array(r[0]) for r in res)
    hess = sum(np.array(r[1]) for r in res)
    ll_new = sum(r[2] for r in res)
    return grad, hess, ll_new


def init_model(n_cols: int, n_obs: int):
    ll = -2 * n_obs * np.log(2)
    coeff = np.zeros(n_cols)
    return coeff, ll


def update_coefficients(grad: np.ndarray, hess: np.ndarray):
    covariance = np.linalg.inv(hess)
    coeff = covariance @ grad
    return coeff


if __name__ == "__main__":
    import time

    s = time.perf_counter()
    logistic_regression()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
