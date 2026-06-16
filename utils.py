from itertools import combinations, product
import math
from scipy.optimize import linprog
from scipy.linalg import null_space
from scipy.stats import ortho_group
import pandas as pd
import random
import glob
import os
import numpy as np

import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm

def number_of_activations(n, d):
    # if d >= n: 2 * 2^n
    # if d < n: 4 * sum_{i=0}^{d-1} (n-1 choose i)
    if d >= n:
        return 2**n
    else:
        total = 0
        for i in range(d):
            total += math.comb(n-1, i)
        return 2 * total
    
def get_all_nonempty_activations(X):
    eps=1e-8

    d = X.shape[1]
    n = X.shape[0]

    if d >= n:
        result = np.array([np.array(list(bin(i)[2:].zfill(n)), dtype=int) for i in range(2**n)])
    else:
        binary_position_values = 2**np.arange(n, dtype=np.int64)[::-1]
        all_activations = set()
        for indices in tqdm(combinations(range(n-1), d-1), total=math.comb(n-1, d-1)):
            selected_rows = X[list(indices), :]
            null_space_vector = null_space(selected_rows).T[0]
            for v in [null_space_vector, -null_space_vector]:
                activation = X @ v
                zero_indices = np.where(np.abs(activation) < eps)[0]
                for zero_pattern in product([-1, 1], repeat=d-1):
                    # Create a copy of the activation
                    act_copy = activation.copy()
                    
                    # Fill in the near-zero entries with all combinations of +1/-1
                    for i, idx in enumerate(zero_indices):
                        act_copy[idx] = zero_pattern[i]
                    
                    # Convert to binary activation pattern (0/1)
                    binary_act = ((np.sign(act_copy) + 1) // 2).astype(int)
                    activation_int = tuple(binary_act)
                    all_activations.add(activation_int)

        result = np.array(list(all_activations), dtype=int)
    
    if len(result) != number_of_activations(n, d):
        raise Exception("Number of activations does not match expected number")
    return result

def merge_neurons(W, angle_tolarance=1e-1, zero_tolerance=1e-6):
    W = W.copy()
    W = W[np.linalg.norm(W, axis=1) > zero_tolerance]

    W_normalized = W / np.linalg.norm(W, axis=1, keepdims=True)
    similarity_matrix = W_normalized @ W_normalized.T

    if similarity_matrix.size == 0:
        return W_normalized, 1
    max_angle = np.min(similarity_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    merged_indices = set()
    for i in range(similarity_matrix.shape[0]):
        if i in merged_indices:
            continue
        similar_rows = np.where(similarity_matrix[i] > np.cos(angle_tolarance))[0]
        if len(similar_rows) > 0:
            merged_indices.update(similar_rows)
            # Sum up the rows that are similar
            W[i] = W[[i] + list(similar_rows)].sum(axis=0)

    # Keep only unique rows
    return W[list(set(range(W.shape[0])) - merged_indices)], max_angle

def generate_activation_approx_sampling(args, X, eps=1e-8):
    d = X.shape[1]
    n = X.shape[0]

    # if d > n, rejection sampling should work well
    if d > n:
        return generate_rejection_sample(X)

    # Pick d-1 random data vectors
    random_indices = np.random.choice(n, d-1, replace=False)
    random_vectors = X[random_indices, :]

    # Find the null space of these vectors
    null_space_vector = null_space(random_vectors).T[0]

    # randomly choose the direction of the null space vector
    null_space_vector = np.random.choice([-1, 1]) * null_space_vector

    activation = X @ null_space_vector
    # set entries with abs value < eps to +1 or -1 at random
    activation[np.abs(activation) < eps] = np.random.choice([-1, 1], size=d - 1)
    activation = (np.sign(activation) + 1) / 2

    return activation

def generate_rw_activation(args, X):
    x = generate_activation_approx_sampling(args, X)

    activation = generate_activation_walk(X, x=x)

    return activation

def generate_binary_combinations_nd(shape):
    num_elements = np.prod(shape)
    binary_combinations = list(product([0, 1], repeat=num_elements))
    numpy_arrays = [np.array(comb).reshape(shape) for comb in binary_combinations]
    return numpy_arrays

def create_data(n, d, label_type='polynomial', m = None):
    # Create nxd data matrix with gaussian entries
    X = np.random.randn(n, d)

    if label_type == 'polynomial':
        coeff = np.random.uniform(-1, 1, 3)
        y = np.sum(np.polynomial.polynomial.polyval(X, coeff), axis=1)
    elif label_type == 'teacher':
        # Create a random teacher network that defines the labels
        W = np.random.randn(m, d)
        alpha = np.random.randn(m)
        first_layer_output = (W @ X.T).T
        #relu
        first_layer_output[first_layer_output < 0] = 0
        y = np.sum(alpha * first_layer_output, axis=1)
    elif label_type == 'random':
        y = np.random.uniform(-1, 1, n)
    elif label_type == 'orthogonal':
        if d < n:
            raise ValueError("d must be greater than or equal to n for orthogonal labels.")

        # Generate orthogonal vectors
        X = ortho_group.rvs(dim=d)
        X = X[:n, :]

        # Create a random teacher network that defines the labels
        W = np.random.randn(m, d)
        alpha = np.random.randn(m)
        first_layer_output = (W @ X.T).T
        #relu
        first_layer_output[first_layer_output < 0] = 0
        y = np.sum(alpha * first_layer_output, axis=1)
    else:
        raise ValueError("Invalid label type. Choose 'polynomial', 'teacher', or 'random'.")

    return X, y

def get_activation(X, W, eps=1e-5):
    # Create a mxn matrix of activations
    activation = np.matmul(W, X.T)

    activation[activation > eps] = 1
    activation[activation <= eps] = 0

    return activation

def is_feasible_activation(X, pattern, run_quick_check=True, use_gurobi=True):
    d = X.shape[1]
    n = X.shape[0]

    M = (X.T * (2*pattern-1)).T

    if run_quick_check:
        avg = np.mean(M, axis=0)
        sides = M @ avg
        # if all entries in M @ avg are positive, return True
        if np.all(sides > 0):
            return True

    if not use_gurobi:
        res = linprog(c=np.zeros(d), A_ub=-M, b_ub=-np.ones(n), bounds=(None, None), method='highs-ds', options={'presolve': False})
        return res.success


    model = gp.Model("positive_product_check")
    # Suppress Gurobi output to console
    model.Params.OutputFlag = 0
    # This can speed things up in certain settings. Needs to be tested with and without
    # in each circumstance to determine the best option
    model.Params.Threads = 1

    x = model.addMVar(shape=M.shape[1], name="x", lb=-GRB.INFINITY, ub=GRB.INFINITY)


    model.addConstr(M @ x >= 1, name="all_entries_positive")

    # Set a dummy objective function (since it's a feasibility problem)
    model.setObjective(0, GRB.MINIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.UNBOUNDED:
        return True
    elif model.Status == GRB.INFEASIBLE:
        return False
    else:
        print(f"Optimization ended with Gurobi status code: {model.Status}")
        raise RuntimeError("Gurobi optimization failed.")


def generate_rejection_sample(X):
    n = X.shape[0]
    while True:
        # Generate a random pattern
        pattern = np.random.randint(0, 2, n)
        res = is_feasible_activation(X, pattern)
        if res:
            return pattern

def generate_activation_random_vec(X):
    d = X.shape[1]
    neuron = np.random.randn(d)

    T_i = np.matmul(X, neuron.T)

    # Set positive activations to 1
    T_i[T_i > 0] = 1
    T_i[T_i <= 0] = 0

    return T_i

def generate_activation_walk(X, x=None, steps=None):
    d = X.shape[1]
    n = X.shape[0]
    if steps is None:
        steps = 2*int(n/d)
    if x is None:
        x = generate_activation_random_vec(X)
        x = x.flatten()

    for i in range(steps):
        # Flip a random bit
        j = random.randint(0, n-1)
        x[j] = 1 - x[j]
        # Check if the new pattern is feasible
        if not is_feasible_activation(X, x):
            x[j] = 1 - x[j]

    return x

def group_by_equality(W_subset, activation):
    """
    Groups the rows of W_subset based on the equality of their corresponding
    rows in the activation matrix.
    """
    groups = {}
    for i, act_row in enumerate(activation.tolist()):  # Convert to list for easy comparison
        act_tuple = tuple(act_row)  # Use tuple as it's hashable for dictionary keys
        if act_tuple not in groups:
            groups[act_tuple] = []
        groups[act_tuple].append(i)
    return groups


def concat_csv_files(args):
    # Get all CSV files in the directory
    pkl_files = glob.glob(os.path.join(args.data_dir, 'prop*.pkl'))
    xy_files = glob.glob(os.path.join(args.data_matrices, '*.pkl'))

    result_data = []
    xy_data = []
    for file in tqdm(pkl_files):
        with open(file, 'rb') as f:
            data = pd.read_pickle(f)
            result_data.extend(data)

    for file in tqdm(xy_files):
        with open(file, 'rb') as f:
            data = pd.read_pickle(f)
            xy_data.append(data)

    # Save as pickle
    with open(f'{args.data_dir}/results_data.pkl', 'wb') as f:
        pd.to_pickle(result_data, f)
    print("Saved results_data.pkl")
    with open(f'{args.data_dir}/xy_data.pkl', 'wb') as f:
        pd.to_pickle(xy_data, f)
    print("Saved xy_data.pkl")

    result_df = pd.DataFrame(result_data)
    xy_data = pd.DataFrame(xy_data)

    print("Created DataFrames")

    # Group by n, d, beta, m, data_matrix_index and count length of each results list and sum
    grouped = result_df.groupby(['n', 'd', 'beta', 'm', 'data_key'])

    # Calculate the length of each 'results' list and the sum of 'results' (if it's numeric)
    def aggregate_results(series):
        all_losses = np.concatenate(series.tolist()) if not series.empty else np.array([])
        total_length = len(all_losses)
        return pd.Series({'num_runs': total_length})
    
    print("Aggregating results...")

    analysis_df = grouped['losses'].apply(aggregate_results).reset_index()

    print("Add xy_data...")

    # Join to xy_data on data_key and add xy_data info
    analysis_df = analysis_df.merge(xy_data, left_on='data_key', right_on="key", how='left')

    print(analysis_df)

    # You can save this analysis_df to a file if needed
    analysis_df.drop(columns=['X', 'y', 'nonempty_activations']).to_csv(f"{args.data_dir}/results_analysis.csv", index=False)

def main():
    print(number_of_activations(5, 2))
    print(number_of_activations(5, 5))
    print(number_of_activations(10, 2))
    print(number_of_activations(10, 5))

if __name__ == "__main__":
    main()
