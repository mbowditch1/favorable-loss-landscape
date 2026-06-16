import numpy as np
import csv
from joblib import Memory
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from convex import solve_convex_subprogram
from nn import test_minimum
from utils import (
    create_data,
    get_all_nonempty_activations,
    merge_neurons,
    generate_binary_combinations_nd,
    generate_activation_random_vec,
    generate_rejection_sample,
    generate_activation_walk,
    generate_activation_approx_sampling,
    generate_rw_activation,
)


def create_datasets(args):
    os.makedirs("data_matrices", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    n_range = args.n.split(",")
    d_range = args.d.split(",")
    beta_range = args.beta_list.split(",")
    n_range = [int(n) for n in n_range]
    d_range = [int(d) for d in d_range]
    beta_range = [float(beta) for beta in beta_range]
    for n in n_range:
        for d in d_range:
            print(f"Creating datasets for n={n}, d={d}")
            for i in tqdm(range(args.N)):
                random_key = np.random.randint(0, 1000000000)
                X, y = create_data(n, d, args.data_gen_method, args.teacher_size)
                nonempty_activations = get_all_nonempty_activations(X)
                optima = [
                    {
                        "beta": beta,
                        "value": find_optimum(X, y, beta, nonempty_activations)[0],
                    }
                    for beta in beta_range
                ]
                file_name = f"{args.data_matrices}/n_{n}_d_{d}_{i}.pkl"
                data_xy = {
                    "key": random_key,
                    "file_name": file_name,
                    "X": X,
                    "y": y,
                    "n": n,
                    "d": d,
                    "index": i,
                    "method": args.data_gen_method,
                    "size": args.teacher_size,
                    "nonempty_activations": nonempty_activations,
                    "optima": optima,
                }

                with open(file_name, "wb") as f:
                    pickle.dump(data_xy, f)


def find_optimum(X, y, beta, nonempty_activations):
    T_matrices = [np.diag(T_i.flatten()) for T_i in nonempty_activations]
    T_matrices.extend([-T for T in T_matrices])
    variables = {}
    variables["T_matrices"] = T_matrices
    variables["m"] = len(nonempty_activations)
    variables["n"] = X.shape[0]
    variables["d"] = X.shape[1]

    W, result = solve_convex_subprogram(variables, X, y, beta)

    return float(result), merge_neurons(W)[0]


def solve_N_convex(args, n, m_range, d, beta, N):
    data_matrix_file_name = (
        f"{args.data_matrices}/n_{n}_d_{d}_{args.data_matrix_index}.pkl"
    )
    # Check if there is data saved
    if not os.path.exists(data_matrix_file_name):
        raise ValueError(
            f"Data matrix file {data_matrix_file_name} does not exist. Please create the datasets first."
        )
    else:
        with open(data_matrix_file_name, "rb") as f:
            print("Using saved matrix")
            data_xy = pickle.load(f)
            X = data_xy["X"]
            y = data_xy["y"]
            nonempty_activations = data_xy["nonempty_activations"]
            optimum_value = next(
                (opt["value"] for opt in data_xy["optima"] if opt["beta"] == beta), None
            )

    data_key = data_xy["key"]
    random_key = np.random.randint(0, 1000000000)
    data = []
    total_convex_programs = len(m_range) * N
    rng = np.random.default_rng()

    with tqdm(total=total_convex_programs, desc="Solving Convex Programs") as pbar:
        for m in m_range:
            losses = []
            weight_matrices = []
            activation_patterns = []
            num_regions_containing_neurons = []
            for _ in range(N):
                activations = []
                if args.sampling_method == "uniform":
                    random_indices = rng.choice(
                        np.arange(nonempty_activations.shape[0]), size=m, replace=True
                    )
                    activations = [nonempty_activations[idx] for idx in random_indices]
                else:
                    for _ in range(m):
                        if args.sampling_method == "rw":
                            activations.append(generate_rw_activation(args, X))
                        elif args.sampling_method == "rejection":
                            activations.append(generate_rejection_sample(X))
                        elif args.sampling_method == "random_walk":
                            activations.append(generate_activation_walk(X))
                        elif args.sampling_method == "rs":
                            activations.append(
                                generate_activation_approx_sampling(args, X)
                            )
                        elif args.sampling_method == "random_vector":
                            activations.append(generate_activation_random_vec(X))
                        else:
                            raise ValueError(
                                f"Unknown sampling method: {args.sampling_method}"
                            )

                # Pick positive or negative direction uniformly at random
                activations = [
                    np.sign(np.random.uniform(-1, 1)) * activation
                    for activation in activations
                ]

                unique_activations = np.unique(np.asarray(activations), axis=0)
                num_regions_containing_neurons.append(len(unique_activations))

                activations = unique_activations
                T_matrices = [np.diag(T_i.flatten()) for T_i in activations]
                variables = {}
                variables["T_matrices"] = T_matrices
                variables["m"] = m
                variables["n"] = n
                variables["d"] = d

                W, result = solve_convex_subprogram(variables, X, y, beta)
                losses.append(float(result))
                has_optimum_value = abs(float(result) - optimum_value) < 2e-9
                # print(f"Has optimum value: {has_optimum_value}")
                weight_matrices.append(W)
                activation_patterns.append(activations)

                pbar.update(1)

            data.append(
                {
                    "random_key": random_key,
                    "data_key": data_key,
                    "m": m,
                    "n": n,
                    "d": d,
                    "N": N,
                    "beta": beta,
                    "sampling_method": args.sampling_method,
                    "losses": np.array(losses),
                    "num_regions_containing_neurons": num_regions_containing_neurons,
                    "weight_matrices": weight_matrices,
                    "activations": activation_patterns,
                }
            )

    with open(f"{args.data_dir}/prop_optimum_approx_{random_key}.pkl", "wb") as f:
        pickle.dump(data, f)


def prop_optimum_approx_plot(args):
    N = args.N
    beta = args.beta

    if type(args.m) is str:
        m_range = [int(m) for m in args.m.split(",")]
    else:
        m_range = [args.m]
    if type(args.d) is str:
        d_range = [int(d) for d in args.d.split(",")]
    else:
        d_range = [args.d]
    if type(args.n) is str:
        n_range = [int(n) for n in args.n.split(",")]
    else:
        n_range = [args.n]

    for n in n_range:
        for d in d_range:
            print(
                f"Running prop_optimum_approx with n={n}, d={d}, m_range={m_range}, N={N}, sampling_method={args.sampling_method}"
            )
            solve_N_convex(args, n, m_range, d, beta, N)


def activation_pattern_distribution(args):
    n = args.n
    d = args.d
    N = args.N
    X, y = create_data(n, d, args.data_gen_method, args.teacher_size)

    if args.all_patterns:
        all_patterns_vec = generate_binary_combinations_nd((n, 1))
        all_patterns = [list(pattern.flatten()) for pattern in all_patterns_vec]
        print(f"Number of activation patterns: {len(all_patterns)}")

        activation_patterns = {
            "".join(str(x) for x in pattern): [0, 0, 0, 0, 0]
            for pattern in all_patterns
        }
    else:
        activation_patterns = {}

    # Do random vector sampling
    for i in tqdm(range(N), desc="Random Vector Sampling"):
        activation = generate_activation_random_vec(X)
        activation = list(activation)
        activation = "".join(str(int(x)) for x in activation)

        # Check if activation is in the dictionary
        if activation not in activation_patterns:
            activation_patterns[activation] = [0, 0, 0, 0, 0]

        activation_patterns[activation][0] += 1

    # Rejection sampling
    if args.rejection_sampling:
        for i in tqdm(range(N), desc="Rejection Sampling"):
            activation = generate_rejection_sample(X)
            activation = list(activation)
            activation = "".join(str(int(x)) for x in activation)

            # Check if activation is in the dictionary
            if activation not in activation_patterns:
                activation_patterns[activation] = [0, 0, 0, 0, 0]

            activation_patterns[activation][1] += 1

    # Rejection sampling
    if args.walk_sampling:
        for i in tqdm(range(N), desc="Random Walk Sampling"):
            activation = generate_activation_walk(X)
            activation = list(activation)
            activation = "".join(str(int(x)) for x in activation)

            # Check if activation is in the dictionary
            if activation not in activation_patterns:
                activation_patterns[activation] = [0, 0, 0, 0, 0]

            activation_patterns[activation][2] += 1

    # RS sampling
    if args.approx_unif_sampling:
        for i in tqdm(range(N), desc="Approx Unif Sampling"):
            activation = generate_activation_approx_sampling(args, X)
            activation = list(activation)
            activation = "".join(str(int(x)) for x in activation)

            # Check if activation is in the dictionary
            if activation not in activation_patterns:
                activation_patterns[activation] = [0, 0, 0, 0, 0]

            activation_patterns[activation][3] += 1
    if args.uniform_sampling:
        activation = get_all_nonempty_activations(X)
        raise ValueError(
            f"Uniform sampling not implemented yet. Please use one of the other sampling methods."
        )

    # RW sampling
    if args.rw_sampling:
        for i in tqdm(range(N), desc="RW Sampling"):
            activation = generate_rw_activation(args, X)
            activation = list(activation)
            activation = "".join(str(int(x)) for x in activation)

            # Check if activation is in the dictionary
            if activation not in activation_patterns:
                activation_patterns[activation] = [0, 0, 0, 0, 0]

            activation_patterns[activation][4] += 1

    # Get counts for each method
    random_vec_counts = [
        activation_patterns[pattern][0] for pattern in activation_patterns
    ]
    rejection_counts = [
        activation_patterns[pattern][1] for pattern in activation_patterns
    ]
    random_walk_counts = [
        activation_patterns[pattern][2] for pattern in activation_patterns
    ]
    rs_counts = [activation_patterns[pattern][3] for pattern in activation_patterns]
    rw_counts = [activation_patterns[pattern][4] for pattern in activation_patterns]
    patterns = list(activation_patterns.keys())

    # Sort by the counts from random vector sampling
    sorted_indices = np.argsort(random_vec_counts)[::-1]  # Sort in descending order
    sorted_patterns = [patterns[i] for i in sorted_indices]
    sorted_random_vec_counts = [random_vec_counts[i] for i in sorted_indices]
    sorted_rejection_counts = [rejection_counts[i] for i in sorted_indices]
    sorted_random_walk_counts = [random_walk_counts[i] for i in sorted_indices]
    sorted_rs_counts = [rs_counts[i] for i in sorted_indices]
    sorted_rw_counts = [rw_counts[i] for i in sorted_indices]

    # Create the figure and axes
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, figsize=(20, 12), sharey=True
    )

    # Add title to fig
    fig.suptitle(f"Activation Pattern Distribution (n={n}, d={d})", fontsize=16)

    # Plot the histogram for Random Vector Sampling
    ax1.bar(
        np.arange(len(sorted_patterns)), sorted_random_vec_counts, label="Random vector"
    )
    ax1.set_xlabel("Activation Pattern")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Random Vector Sampling")
    ax1.set_xticklabels([])  # Remove x-ticks

    # Plot the histogram for rejection sampling
    ax2.bar(
        np.arange(len(sorted_patterns)),
        sorted_rejection_counts,
        label="Rejection sampling",
    )
    ax2.set_xlabel("Activation Pattern")
    ax2.set_title("Rejection Sampling")
    ax2.set_xticklabels([])  # Remove x-ticks

    # Plot the histogram for rejection sampling
    ax3.bar(
        np.arange(len(sorted_patterns)), sorted_random_walk_counts, label="Random walk"
    )
    ax3.set_xlabel("Activation Pattern")
    ax3.set_title("Random Walk Sampling")
    ax3.set_xticklabels([])  # Remove x-ticks

    # Plot the histogram for rejection sampling
    ax4.bar(np.arange(len(sorted_patterns)), sorted_rs_counts, label="RS sampling")
    ax4.set_xlabel("Activation Pattern")
    ax4.set_title("RS Sampling")
    ax4.set_xticklabels([])  # Remove x-ticks

    # Plot the histogram for rw sampling
    ax5.bar(np.arange(len(sorted_patterns)), sorted_rw_counts, label="RW sampling")
    ax5.set_xlabel("Activation Pattern")
    ax5.set_title("RW Sampling")
    ax5.set_xticklabels([])  # Remove x-ticks

    plt.tight_layout()
    plt.savefig(f"{args.run_dir}/activation_pattern_distribution.png")

    # Save data as np file
    np.save(f"{args.run_dir}/activation_patterns.npy", activation_patterns)


def plot_histogram(args, n, d, m, losses, optima, data_key):
    diff_to_optima = np.abs(losses - optima)

    # Define log-spaced bins
    num_bins = 100
    log_min = -15
    log_max = 1

    log_bins = np.logspace(log_min, log_max, num=num_bins)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.hist(diff_to_optima, bins=log_bins, edgecolor="black")
    plt.xscale("log")
    plt.title(f"n={n}, d={d}, m={m}, {data_key}")
    plt.xlabel("Distance to Optimum")
    plt.ylabel("Frequency")
    # Draw vertical line at 1e-8
    plt.axvline(x=1e-8, color="r", linestyle="--", label="1e-8")
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{args.run_dir}/n_{n}_d_{d}_m_{m}_{data_key}_histogram.png", dpi=300)
    plt.close()


def plot_prop_optimal(args):
    # load the precomputed table of (n,d,m,beta,data_key,losses,stationary,global_min_loss)
    with open(f"{args.data_dir}/minimum_test.pkl", "rb") as f:
        min_data = pickle.load(f)
    df = pd.DataFrame(min_data)

    with open(f"{args.data_dir}/xy_data.pkl", "rb") as f_xy:
        xy_data = pickle.load(f_xy)

    df_xy = pd.DataFrame(xy_data)

    # Data generation method
    data_gen_method = df_xy["method"].unique()[0]

    for beta in sorted(df["beta"].unique()):
        df_b = df[df["beta"] == beta]

        n_vals = sorted(df_b["n"].unique())
        d_vals = sorted(df_b["d"].unique())

        fig, axes = plt.subplots(
            len(n_vals),
            len(d_vals),
            figsize=(5 * len(d_vals), 4 * len(n_vals)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for i, d_val in enumerate(d_vals):
            for j, n_val in enumerate(n_vals):
                ax = axes[j, i]
                cell = df_b[(df_b["n"] == n_val) & (df_b["d"] == d_val)]

                m_vals = sorted(cell["m"].unique())

                data_keys = cell["data_key"].unique()

                # Iterate through each unique data_key
                props_opt_overall = []
                props_min_overall = []
                mean_num_regions_overall = []
                total_nonempty_activations = []
                for data_key in data_keys:
                    subset_df_key = cell[cell["data_key"] == data_key]

                    if not subset_df_key.empty:
                        props_opt = []
                        props_min = []
                        mean_num_regions = []
                        for m_val in m_vals:
                            subset_df_key_m = subset_df_key[subset_df_key["m"] == m_val]

                            if not subset_df_key_m.empty:
                                # Get the losses and stationary values for this key
                                losses = subset_df_key_m["losses"].values
                                stationary = subset_df_key_m["stationary"].values
                                global_min_loss = subset_df_key_m[
                                    "global_min_loss"
                                ].values
                                prop_regions = subset_df_key_m[
                                    "mean_num_regions"
                                ].values
                                mean_num_regions.append(prop_regions)

                                # Concatenate the losses and stationary values
                                losses = np.concatenate(losses)
                                stationary = np.concatenate(stationary).astype(bool)
                                gmin = global_min_loss[0]

                                plot_histogram(
                                    args, n_val, d_val, m_val, losses, gmin, data_key
                                )

                                # prop of global-optimal
                                p_opt = np.mean(losses < gmin + args.eps)
                                props_opt.append(p_opt)

                                # prop of local minima *excluding* global optima
                                mask = losses >= gmin + args.eps

                                p_min = (
                                    sum(stationary[mask]) / len(losses)
                                    if mask.any()
                                    else 0.0
                                )

                                props_min.append(p_min)

                        props_opt_overall.append(props_opt)
                        props_min_overall.append(props_min)
                        mean_num_regions_overall.append(mean_num_regions)

                        # Get non empty activations
                        xy_data = df_xy[df_xy["key"] == data_key]
                        nonempty_activations = (
                            len(xy_data.iloc[0]["nonempty_activations"]) * 2
                        )
                        total_nonempty_activations.append(nonempty_activations)

                if len(props_opt_overall) == 0:
                    print(f"No data for n={n_val}, d={d_val}")
                    continue

                for p_val in props_opt_overall:
                    print(f"Length of prop_opt: {len(p_val)}")
                print(f"n={n_val}, d={d_val}")
                print(f"props_opt_overall: {props_opt_overall}")
                print(f"m_vals: {m_vals}")

                # now summarize across data_keys
                opt_means = np.mean(props_opt_overall, axis=0)
                opt_stds = np.std(props_opt_overall, axis=0)
                opt_max = np.max(props_opt_overall, axis=0)
                opt_mins = np.min(props_opt_overall, axis=0)
                min_means = np.mean(props_min_overall, axis=0)
                min_stds = np.std(props_min_overall, axis=0)
                min_max = np.max(props_min_overall, axis=0)
                min_mins = np.min(props_min_overall, axis=0)

                region_means = np.mean(mean_num_regions_overall, axis=0)
                region_max = np.max(mean_num_regions_overall, axis=0)
                region_mins = np.min(mean_num_regions_overall, axis=0)
                region_stds = np.std(mean_num_regions_overall, axis=0)

                mean_non_empty_activations = np.mean(total_nonempty_activations)

                # Create CSV file
                # Check if directory exists, if not create it
                if not os.path.exists(args.csv_dir):
                    os.makedirs(args.csv_dir)

                csv_filename = f"{args.csv_dir}/beta_{beta}_n_{n_val}_d_{d_val}_{data_gen_method}.csv"
                with open(csv_filename, "w", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    # Write header
                    csv_writer.writerow(
                        [
                            "m",
                            "prop_optimal_mean",
                            "prop_optimal_max",
                            "prop_optimal_min",
                            "prop_optimal_std",
                            "prop_minima_mean",
                            "prop_minima_max",
                            "prop_minima_min",
                            "prop_minima_std",
                            "prop_non_empty_mean",
                            "prop_non_empty_max",
                            "prop_non_empty_min",
                            "prop_non_empty_std",
                            "total_non_empty",
                        ]
                    )

                    # Write data
                    for (
                        m_val,
                        opt_mean,
                        opt_max,
                        opt_min,
                        opt_std,
                        min_mean,
                        min_max,
                        min_min,
                        min_std,
                        region_mean,
                        region_max,
                        region_min,
                        region_std,
                    ) in zip(
                        m_vals,
                        opt_means,
                        opt_max,
                        opt_mins,
                        opt_stds,
                        min_means,
                        min_max,
                        min_mins,
                        min_stds,
                        region_means,
                        region_max,
                        region_mins,
                        region_stds,
                    ):
                        csv_writer.writerow(
                            [
                                m_val,
                                opt_mean,
                                opt_max,
                                opt_min,
                                opt_std,
                                min_mean,
                                min_max,
                                min_min,
                                min_std,
                                region_mean[0],
                                region_max[0],
                                region_min[0],
                                region_std[0],
                                mean_non_empty_activations,
                            ]
                        )

                # plot
                ax.plot(m_vals, opt_means, label="Global-optimum")
                ax.fill_between(
                    m_vals,
                    np.array(opt_means) - np.array(opt_stds),
                    np.array(opt_means) + np.array(opt_stds),
                    alpha=0.2,
                )

                ax.plot(m_vals, min_means, label="Local minima")
                ax.fill_between(
                    m_vals,
                    np.array(min_means) - np.array(min_stds),
                    np.array(min_means) + np.array(min_stds),
                    alpha=0.2,
                )

                # Plot mean number of regions as dotted line
                ax.plot(
                    m_vals,
                    region_means,
                    label="Proportion of nonempty regions with a neuron",
                    linestyle="dotted",
                )

                # Plot vertical line at total_nonempty_activations
                ax.axvline(
                    x=mean_non_empty_activations,
                    color="red",
                    linestyle="--",
                    label="Total nonempty activations",
                )

                ax.set_title(f"n={n_val}, d={d_val}")
                ax.grid(True)
                if j == len(n_vals) - 1:
                    ax.set_xscale("log")
                    ax.set_xlabel("m")
                if i == 0:
                    ax.set_ylabel("Proportion")

                if j == 0 and i == 0:  # Only collect handles once
                    handles, labels = ax.get_legend_handles_labels()

        fig.suptitle(f"Proportion Optimal vs. Local Minima (β={beta})", fontsize=16)
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Increased bottom margin
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.00), ncol=2
        )
        plt.savefig(f"{args.run_dir}/prop_optimum_global_min_beta_{beta}.png")
        plt.close(fig)


def get_prop_optimal(args):
    # create a disk cache in data_dir (for caching test_minimum)
    cache = Memory(location=f"{args.data_dir}/stationary_cache", verbose=0)

    @cache.cache
    def test_minimum_cached(X, y, W, beta, activations, cvx_solver):
        return test_minimum(args, X, y, W, beta, activations, cvx_solver=cvx_solver)

    with open(f"{args.data_dir}/results_data.pkl", "rb") as f:
        result_data = pickle.load(f)

    with open(f"{args.data_dir}/xy_data.pkl", "rb") as f_xy:
        xy_data = pickle.load(f_xy)

    result_df = pd.DataFrame(result_data)
    xy_df = pd.DataFrame(xy_data)

    beta_range = sorted(result_df["beta"].unique())
    output_rows = []

    for beta in beta_range:
        beta_subset_df = result_df[result_df["beta"] == beta]

        # Extract unique values of n, m, and d from the DataFrame
        n_range = sorted(beta_subset_df["n"].unique())
        m_range = sorted(beta_subset_df["m"].unique())
        d_range = sorted(beta_subset_df["d"].unique())

        num_n = len(n_range)
        num_d = len(d_range)

        # Create the figure and subplots
        fig, axes = plt.subplots(
            num_n, num_d, figsize=(5 * num_d, 4 * num_n), sharex=True, sharey=True
        )

        # Ensure axes is a 2D array even if num_n or num_d is 1
        if num_n == 1 and num_d == 1:
            axes = np.array([[axes]])
        elif num_n == 1:
            axes = axes[np.newaxis, :]
        elif num_d == 1:
            axes = axes[:, np.newaxis]

        # Iterate through each combination of n and d
        for i, d_val in enumerate(tqdm(d_range)):
            for j, n_val in enumerate(n_range):
                ax = axes[j, i]

                # Filter data for the current n and d
                subset_df = beta_subset_df[
                    (beta_subset_df["n"] == n_val) & (beta_subset_df["d"] == d_val)
                ]

                print(f"Plotting n={n_val}, d={d_val}")

                # Get unique data_keys
                unique_keys = subset_df["data_key"].unique()

                proportion_optimal = []
                proportion_minima = []

                # Iterate through each unique data_key
                for key in unique_keys:
                    subset_df_key = subset_df[subset_df["data_key"] == key]
                    print(f"Key: {key}")

                    if not subset_df_key.empty:
                        # Get X and y data for this key
                        xy_data = xy_df[xy_df["key"] == key]

                        # If empty, skip this key
                        if xy_data.empty:
                            print(
                                f"Warning: No matching data found in xy_data for key: {key}"
                            )
                            continue

                        X = xy_data.iloc[0]["X"]
                        y = xy_data.iloc[0]["y"]
                        nonempty_activations = (
                            len(xy_data.iloc[0]["nonempty_activations"]) * 2
                        )

                        # Find optima that matches beta
                        optimum = xy_data.iloc[0]["optima"]
                        for opt in optimum:
                            if opt["beta"] == beta:
                                optima = opt["value"]
                                break

                        # Extract results and m values
                        loss_list = subset_df_key["losses"].tolist()
                        m_values = subset_df_key["m"].tolist()
                        weight_matrices = subset_df_key["weight_matrices"].tolist()
                        activations = subset_df_key["activations"].tolist()
                        num_regions_containing_neurons = subset_df_key[
                            "num_regions_containing_neurons"
                        ].tolist()

                        # Ensure m_values and loss_list are aligned
                        sorted_indices = np.argsort(m_values)
                        sorted_m_values = np.array(m_values)[sorted_indices]
                        sorted_loss_list = [loss_list[k] for k in sorted_indices]
                        sorted_weight_matrices = [
                            weight_matrices[k] for k in sorted_indices
                        ]
                        sorted_activations = [activations[k] for k in sorted_indices]
                        sorted_num_regions = [
                            num_regions_containing_neurons[k] for k in sorted_indices
                        ]

                        # Concatenate any losses that have the same m value
                        unique_m_values = np.unique(sorted_m_values)
                        unique_loss_list = []
                        unique_weight_list = []
                        unique_activation_list = []
                        unique_num_regions_list = []
                        for m_val in unique_m_values:
                            losses_for_m = [
                                loss
                                for m, loss in zip(sorted_m_values, sorted_loss_list)
                                if m == m_val
                            ]
                            weights_for_m = [
                                weights
                                for m, weights in zip(
                                    sorted_m_values, sorted_weight_matrices
                                )
                                if m == m_val
                            ]
                            activations_for_m = [
                                act
                                for m, act in zip(sorted_m_values, sorted_activations)
                                if m == m_val
                            ]
                            num_regions_for_m = [
                                num
                                for m, num in zip(sorted_m_values, sorted_num_regions)
                                if m == m_val
                            ]

                            # Flatten the lists
                            weights_for_m = [x for xs in weights_for_m for x in xs]
                            activations_for_m = [
                                x for xs in activations_for_m for x in xs
                            ]
                            num_regions_for_m = [
                                x for xs in num_regions_for_m for x in xs
                            ]
                            if losses_for_m:
                                unique_loss_list.append(np.concatenate(losses_for_m))
                                unique_weight_list.append(weights_for_m)
                                unique_activation_list.append(activations_for_m)
                                unique_num_regions_list.append(num_regions_for_m)

                        # Loop through losses and check if local minima
                        for (
                            W_for_m,
                            losses_for_m,
                            m_val,
                            act_for_m,
                            region_for_m,
                        ) in zip(
                            unique_weight_list,
                            unique_loss_list,
                            unique_m_values,
                            unique_activation_list,
                            unique_num_regions_list,
                        ):
                            # Check if each loss for this m is within eps of the global minimum
                            stationary_points = []
                            for W_i, loss_i, act_i in zip(
                                W_for_m, losses_for_m, act_for_m
                            ):
                                # Check if the loss is within eps of the global minimum
                                if loss_i >= optima + args.eps:
                                    stationary = test_minimum_cached(
                                        X, y, W_i, beta, act_i, cvx_solver="cvxpy"
                                    )
                                else:
                                    stationary = True

                                stationary_points.append(stationary)

                            prop_num_regions = (
                                np.array(region_for_m) / nonempty_activations
                            )

                            # Add to dataset
                            output_rows.append(
                                {
                                    "n": n_val,
                                    "d": d_val,
                                    "m": m_val,
                                    "beta": beta,
                                    "data_key": key,
                                    "stationary": stationary_points,
                                    "prop_num_regions": prop_num_regions,
                                    "mean_num_regions": np.mean(prop_num_regions),
                                    "losses": losses_for_m,
                                    "global_min_loss": optima,
                                }
                            )
                            # Save as pickle
                            with open(f"{args.data_dir}/minimum_test.pkl", "wb") as f:
                                pickle.dump(output_rows, f)

                            # Save as csv also
                            df_output = pd.DataFrame(output_rows)
                            df_output.to_csv(
                                f"{args.data_dir}/minimum_test.csv", index=False
                            )

    # Save as pickle
    with open(f"{args.data_dir}/minimum_test.pkl", "wb") as f:
        pickle.dump(output_rows, f)

    # Save as csv also
    df_output = pd.DataFrame(output_rows)
    df_output.to_csv(f"{args.data_dir}/minimum_test.csv", index=False)
