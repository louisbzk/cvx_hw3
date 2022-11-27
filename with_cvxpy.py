import numpy as np
import cvxpy as cvx
from typing import Tuple

N_SAMPLES = 50
DIM = 300
LAMBDA = 10.
DATA_SCALE = 10.
np.random.seed(1)


def generate_random_data(n_samples, dim, scale, w_density, y_dev) -> Tuple[np.ndarray, np.ndarray]:
    X = scale * 2 * (np.random.rand(n_samples, dim) - 0.5)
    w_star = np.random.randn(dim)
    idx_nonzero = np.random.choice(range(dim), int((1 - w_density) * dim), replace=False)
    for i in idx_nonzero:
        w_star[i] = 0.
    y = X @ w_star + np.random.normal(0, y_dev, size=n_samples)

    return X, y


def main():
    X, y = generate_random_data(N_SAMPLES, DIM, DATA_SCALE, w_density=0.4, y_dev=0.5)

    Q = 0.5 * np.eye(N_SAMPLES)
    p = -y
    A = np.vstack([X.T, -X.T])
    b = np.full(shape=2 * DIM, fill_value=LAMBDA)

    # Create dual
    v = cvx.Variable(shape=N_SAMPLES, name='v')
    _Q = cvx.Parameter(shape=Q.shape, name='Q', PSD=True)
    _Q.value = Q
    _p = cvx.Parameter(shape=p.shape, name='p')
    _p.value = p
    dual_constraints = [A @ v <= b]
    dual_objective = cvx.Minimize(cvx.quad_form(x=v, P=_Q) + _p.T @ v)

    dual_problem = cvx.Problem(dual_objective, dual_constraints)
    dual_problem.solve()
    print(f'Status : {dual_problem.status}\n'
          f'Optimal value : {dual_problem.value}\n')

    # Create primal
    w = cvx.Variable(shape=DIM, name='w')
    w.value = np.zeros(shape=DIM, dtype=float)
    lambd = cvx.Parameter(nonneg=True, name='lambda')
    lambd.value = LAMBDA
    primal_objective = cvx.Minimize(
        cvx.Constant(0.5) * cvx.norm2(X @ w - y)**2 + lambd * cvx.norm1(w)
    )

    primal_problem = cvx.Problem(primal_objective)
    primal_problem.solve()
    print(f'Status : {primal_problem.status}\n'
          f'Optimal value : {primal_problem.value}\n')


if __name__ == '__main__':
    main()
