from experiment import prop_optimum_approx_plot, activation_pattern_distribution, plot_prop_optimal, create_datasets, get_prop_optimal
from utils import concat_csv_files
import parse_args
import shutil
import sys
from datetime import datetime
import os

def main(args):
    if args.prop_optimum_approx:
        prop_optimum_approx_plot(args)
    elif args.activation_pattern_distribution:
        activation_pattern_distribution(args)
    elif args.concat_csv:
        concat_csv_files(args)
    elif args.plot_prop_optimal:
        plot_prop_optimal(args)
    elif args.create_datasets:
        create_datasets(args)
    elif args.get_prop_optimal:
        get_prop_optimal(args)

if __name__ == "__main__":
    args = parse_args.make(*sys.argv[1:])

    # Get timestamp and format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create run directory
    args.run_dir = f"runs/{args.script_name.replace('.sh', '')}_{timestamp}"
    os.makedirs(args.run_dir, exist_ok=True)

    # Copy script to run directory
    args.script_name = f"scripts/{args.script_name}"
    shutil.copy(args.script_name, args.run_dir)

    main(args)
