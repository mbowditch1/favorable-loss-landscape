import argparse

def make(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_name", default="sanity_check.sh", type=str, help="")
    parser.add_argument("--data_dir", default="data", type=str, help="")
    parser.add_argument("--csv_dir", default="csv", type=str, help="")
    parser.add_argument("--data_matrices", default="data_matrices", type=str, help="")

    parser.add_argument("--m", default=10, help="")
    parser.add_argument("--d", default=3, help="")
    parser.add_argument("--n", default=10, help="")

    parser.add_argument("--data_gen_method", default="teacher", type=str, help="")
    parser.add_argument("--teacher_size", default=10, type=int, help="")

    parser.add_argument("--beta", default=0.01, type=float, help="")
    parser.add_argument("--num_iter", default=100, type=int, help="")

    parser.add_argument("--activation_pattern_distribution", action="store_true", help="")
    parser.add_argument("--plot_prop_optimal", action="store_true", help="")
    parser.add_argument("--get_prop_optimal", action="store_true", help="")
    parser.add_argument("--plot_prop_minimum", action="store_true", help="")
    parser.add_argument("--concat_csv", action="store_true", help="")
    parser.add_argument("--create_datasets", action="store_true", help="")
    parser.add_argument("--prop_optimum_approx", action="store_true", help="")

    parser.add_argument("--all_patterns", action="store_true", help="")
    parser.add_argument("--rejection_sampling", action="store_true", help="")
    parser.add_argument("--walk_sampling", action="store_true", help="")
    parser.add_argument("--approx_unif_sampling", action="store_true", help="")
    parser.add_argument("--rw_sampling", action="store_true", help="")
    parser.add_argument("--sampling_method", default="random_vector", type=str, help="")

    parser.add_argument("--n_interval", default=2, type=int, help="")
    parser.add_argument("--m_interval", default=2, type=int, help="")

    parser.add_argument("--N", default=100, type=int, help="")
    parser.add_argument("--eps", default=1e-4, type=float, help="")
    parser.add_argument("--relu_tol", default=5e-5, type=float, help="")
    parser.add_argument("--deriv_tol", default=5e-5, type=float, help="")
    parser.add_argument("--convert_tol", default=5e-5, type=float, help="")

    parser.add_argument("--n_list", type=str, help="")
    parser.add_argument("--d_list", type=str, help="")
    parser.add_argument("--beta_list", default="0.01", type=str, help="")

    parser.add_argument("--data_matrix_index", type=int, help="")

    return parser.parse_args(args)
