import cvxpy as cp
from utils import get_activation
import numpy as np
import scipy.sparse as sp

def get_m_t_matrices(X, m):
    d = X.shape[1]
    n = X.shape[0]
    variables = {}
    T_matrices = []

    while len(T_matrices) < m:
        # Create random neuron
        W = np.random.uniform(-1/np.sqrt(m), 1/np.sqrt(m), (1, d))
        T_i = get_activation(X, W)
        if len(T_matrices) == 0:
            T_matrices.append(T_i)
            continue
        is_in_list = np.any(np.all(T_i == T_matrices, axis=1))
        if is_in_list:
            continue
        else:
            T_matrices.append(T_i)

    T_matrices = [np.diag(T_i.flatten()) for T_i in T_matrices]

    variables['T_matrices'] = T_matrices
    variables['m'] = m
    variables['n'] = n
    variables['d'] = d

    return variables

def get_convex_subprogram(X, W, full=False):
    d = X.shape[1]
    n = X.shape[0]
    variables = {}

    # Create T_j matrix
    if full:
        T_matrices = construct_T_matrices(X)
        m = len(T_matrices)
    else:
        m = W.shape[0]
        activation = get_activation(X, W)
        T_matrices = []
        for i in range(m):
            T_i = np.zeros(n)
            for j in range(n):
                T_i[j] = activation[i, j]
            if len(T_matrices) == 0:
                T_matrices.append(T_i)
                continue
            is_in_list = np.any(np.all(T_i == T_matrices, axis=1))
            if is_in_list:
                continue
            else:
                T_matrices.append(T_i)

        T_matrices = [np.diag(T_i.flatten()) for T_i in T_matrices]

    variables['T_matrices'] = T_matrices
    variables['m'] = m
    variables['n'] = n
    variables['d'] = d

    return variables

def cvx_constraints_fn(X, W, T_matrices):
    diags = np.stack([np.diag(T) for T in T_matrices], axis=1)
    mask = sp.csr_matrix(np.where(abs(diags) > 0, 1, -1))
    XW = X @ W.T

    return [cp.multiply(mask, XW) >= 0]

def solve_convex_subprogram(variables, X, y, beta, debug=False):
    T_matrices = variables['T_matrices']
    m = len(T_matrices)
    n = X.shape[0]
    d = X.shape[1]

    # Construct the problem.
    W = cp.Variable((m, d))
    objective = cp.Minimize(cvx_objective_fn(X, y, W, beta, T_matrices))
    constraints = cvx_constraints_fn(X, W, T_matrices)

    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        print("Solver failed")
        return None, None

    if debug:
        output = np.zeros(n)
        for i in range(m):
            curr = np.matmul(T_matrices[i], X)
            curr = np.matmul(curr, W.value[i, :])
            output += curr
        print(f"Output of convex solver : {output}")

    if debug:
        # Calculate MSE loss
        mse_loss = np.sum((output - y) ** 2)
        print(f"Convex MSE loss: {mse_loss}")

    return W.value, result


def cvx_objective_fn(X, y, W, beta, T_matrices): 
    n, d = X.shape
    m = len(T_matrices)

    diff_W_flat = cp.reshape(cp.transpose(W), (m * d,))

    weighted_X = np.hstack([T @ X for T in T_matrices])

    z = weighted_X @ diff_W_flat

    residual = z - cp.reshape(y, (n,))
    loss = cp.sum_squares(residual) / n

    reg = cp.sum(cp.norm(W, axis=1))

    return loss + (2 * beta * reg)
