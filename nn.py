import numpy as np
from utils import get_activation, group_by_equality
import matplotlib.pyplot as plt
import gurobipy as gp
import cvxpy as cp

def create_neural_network(m, d):
    # Create mxd matrix with uniform entries -1/sqrt(m) to 1/sqrt(m)
    W = np.random.uniform(-1/np.sqrt(m), 1/np.sqrt(m), (m, d))
    alpha = np.random.uniform(-1/np.sqrt(m), 1/np.sqrt(m), m)

    return W, alpha

def evaluate_neural_network(X, W, alpha):
    # Use ReLU activation function
    for i in range(X.shape[0]):
        x = X[i, :]
        output = np.matmul(alpha, np.maximum(0, np.matmul(W, x.T)))
        print(f"Output for sample {i}: {output}")

def convert_to_minimal(X, W, act, eps=1e-4):
    W = np.where(np.abs(W) < eps, 0, W)
    alpha = np.array(np.zeros((1, W.shape[0])))

    activations = get_activation(X, W)
    for i in range(len(activations)):
        activations[i] *= (1 if np.max(act[i]) > 0 else -1)

    # Group W based on activations
    groups = group_by_equality(W, activations) 
    new_W = np.zeros_like(W)

    for i, (key, value) in enumerate(groups.items()):
        group = value
        W_i = np.sum(W[group], axis=0)
        W_i_norm = np.sqrt(np.linalg.norm(W_i))
        if W_i_norm > eps:
            new_W[i, :] = W_i / W_i_norm
            alpha[0][i] = W_i_norm * (1 if max(key) > 0 else -1)

    return new_W, alpha

def test_min_gurobi(args, X, y, W, beta, alpha, residual):
    tolerance = args.relu_tol
    gradient_model = gp.Model("gradient_model")
    gradient_model.setParam("OutputFlag", 0)  # Suppress output

    gradient_factor = gradient_model.addMVar(
        shape=(X.shape[0], W.shape[0]), name="gradient_factor", lb=1, ub=1, vtype=gp.GRB.CONTINUOUS
    )
    input_vector_weighting = residual * alpha * 2 / len(y)
    mask = (W @ X.T).T < -tolerance
    # where mask is True, set gradient_factor upper bound to 0
    gradient_factor[mask].LB = 0
    gradient_factor[mask].UB = 0
    close_to_zero_mask = np.abs((W @ X.T).T) < tolerance
    gradient_factor[close_to_zero_mask].LB = 0

    resulting_gradient = (X.T @ (input_vector_weighting * gradient_factor)) + (beta * 2 * W.T)
    objective_function = (resulting_gradient * resulting_gradient).sum()
    gradient_model.setObjective(objective_function, gp.GRB.MINIMIZE)
    gradient_model.optimize()
    if gradient_model.status == gp.GRB.OPTIMAL:
        M_solution = gradient_factor.X
        first_layer_gradient = (X.T @ (input_vector_weighting * M_solution)) + (beta * 2 * W.T)
    elif gradient_model.status == gp.GRB.INFEASIBLE:
        raise Exception("Model is infeasible")
    elif gradient_model.status == gp.GRB.UNBOUNDED:
        raise Exception("Model is unbounded")

    return first_layer_gradient

def test_min_cvxpy(args, X, y, W, beta, alpha, residual):
    tolerance = args.relu_tol
    n_samples, n_features = X.shape
    n_outputs, n_hidden = W.shape

    gradient_factor = cp.Variable((n_samples, n_outputs), name="gradient_factor")

    input_vector_weighting = residual * alpha * 2 / len(y)

    activation = (W @ X.T).T
    mask = activation < -tolerance
    close_to_zero_mask = np.abs(activation) < tolerance

    # create a n_samplex times n_outputs matrix filled with 1
    upper_bounds = np.ones((n_samples, n_outputs))
    lower_bounds = np.ones((n_samples, n_outputs))
    upper_bounds[mask] = 0
    lower_bounds[mask] = 0
    lower_bounds[close_to_zero_mask] = 0
    constraints = [gradient_factor >= lower_bounds, gradient_factor <= upper_bounds]

    # Calculate the resulting gradient
    resulting_gradient = (X.T @ (cp.multiply(input_vector_weighting, gradient_factor))) + (beta * 2 * W.T)

    # Define the objective function (sum of squares of the resulting gradient)
    objective_function = cp.sum_squares(resulting_gradient)

    # Define the CVXPY problem
    problem = cp.Problem(cp.Minimize(objective_function), constraints)

    # Solve the problem
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            M_solution = gradient_factor.value
            first_layer_gradient = (X.T @ (np.multiply(input_vector_weighting, M_solution))) + (beta * 2 * W.T)
            return first_layer_gradient
        elif problem.status == cp.INFEASIBLE or problem.status == cp.INFEASIBLE_INACCURATE:
            raise Exception("Problem is infeasible (CVXPY)")
        elif problem.status == cp.UNBOUNDED:
            raise Exception("Problem is unbounded (CVXPY)")
        else:
            print(f"Problem status: {problem.status}")
            return None
    except cp.SolverError as e:
        print(f"Solver error: {e}")
        return None

def test_minimum(args, X, y, W, beta, activations, alpha=None, convert_to_min=True, loss_type='full', cvx_solver='gurobi'):
    if convert_to_min:
        W, alpha = convert_to_minimal(X, W, activations, eps=args.convert_tol)
    else:
        if alpha is None:
            print("Error: alpha is None, but convert_to_minimal is False.")
            return

    relu_at_zero = np.any(np.abs(W @ X.T) < args.relu_tol)

    first_layer_output = np.maximum(W @ X.T, 0)
    output = alpha @ first_layer_output
    residual = output - y
    grads = residual * first_layer_output * 2 / X.shape[0]
    
    
    G = first_layer_output > 0
    
    first_layer_gradient = np.zeros_like(W)
    second_layer_gradient = np.zeros_like(alpha)
    if loss_type == 'mse' or loss_type == 'full':
        first_layer_gradient = (G * (2 * alpha.T @ residual)) @ X / X.shape[0]
        second_layer_gradient = np.sum(grads, axis=1, keepdims=True).T
    if loss_type == 'l2' or loss_type == 'full':
        first_layer_gradient += 2 * beta * W
        second_layer_gradient += 2 * beta * alpha

    at_critical_point = True
    tolerance = args.deriv_tol

    if not np.all(np.abs(second_layer_gradient) < tolerance):
        at_critical_point = False
    else:
        if relu_at_zero:
            if cvx_solver == 'gurobi':
                first_layer_gradient = test_min_gurobi(args, X, y, W, beta, alpha, residual.T)
            elif cvx_solver == 'cvxpy':
                first_layer_gradient = test_min_cvxpy(args, X, y, W, beta, alpha, residual.T)

        # Check if solver failed
        if first_layer_gradient is None:
            at_critical_point = False
        elif not np.all(np.abs(first_layer_gradient) < tolerance):
            at_critical_point = False

    return at_critical_point
