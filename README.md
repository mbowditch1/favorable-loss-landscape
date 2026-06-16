---

# 📘 *Favorability of Loss Landscape with Regularization Requires Both Large Overparametrization and Initialization*

---

## 📦 Installation

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

---

## 🧪 Running Experiments

Experiments for Theorem 2 are run using python3 theorem2.py. Results are logged in Weights & Biases (edit theorem2.py to change your W&B username and the desired project name). The main options are:

| Argument            | Description                                         | Example         |
| ------------------- | --------------------------------------------------- | --------------- |
| `--alpha`           | Initialization scale                                | `0.125`         |
| `--d`               | Dimension of each data point (must be at least 3)   | `3`             |

All other options default to the values used in the paper.


For all other experiments, the experimental pipeline consists of **five stages**, each corresponding to a script. Run them in the order below.

---

### 🔹 Step 1: Create Datasets

```bash
cd scripts
bash create_datasets.sh
```

Generates datasets with varying number and dimension of data points. Configurable options:

| Argument            | Description                                         | Example         |
| ------------------- | --------------------------------------------------- | --------------- |
| `--n`               | Number of data points                               | `"4,6"`         |
| `--d`               | Dimension of each data point                        | `"8,10"`        |
| `--N`               | Number of datasets per (n, d) setting               | `5`             |
| `--data_gen_method` | Method to generate data (`orthogonal` or `teacher`) | `orthogonal`    |
| `--teacher_size`    | Width of teacher network                            | `10`            |
| `--data_matrices`   | Output directory for datasets                       | `data_matrices` |

---

### 🔹 Step 2: Run Experiments

```bash
cd scripts
bash prop_optimum.sh
```

Runs the main optimality experiments. Key arguments:

| Argument              | Description                                               | Example            |
| --------------------- | --------------------------------------------------------- | ------------------ |
| `--n`, `--m`, `--d`   | Accept integers or comma-separated lists for ranges       | `"4,6"`, `"10,20"` |
| `--sampling_method`   | Sampling method for activation patterns (e.g., `uniform`) | `uniform`          |
| `--data_matrix_index` | Index of the dataset to use                               | `0`                |
| `--beta`              | Weight decay regularization parameter                     | `0.01`             |

You can run this script multiple times with different values — results will be collected together later.

---

### 🔹 Step 3: Collate Datasets

```bash
cd scripts
bash concat_csv.sh
```

Combines all completed experiment outputs into a single file.

---

### 🔹 Step 4: Check Stationarity of Solutions

```bash
cd scripts
bash get_prop_optimal.sh
```

Evaluates stationarity and optimality of solutions from experiments. Important tolerances:

| Argument        | Description                              | Default Value |
| --------------- | ---------------------------------------- | ------------- |
| `--eps`         | Distance to global minima threshold      | `1e-7`        |
| `--relu_tol`    | Tolerance for ReLU derivative          | `5e-5`        |
| `--deriv_tol`   | Tolerance for determining 0 gradient    | `5e-5`        |

---

### 🔹 Step 5: Plot Results

```bash
cd scripts
bash plot_prop_optimal.sh
```

Generates plots to visualize experiment results. These can be found in the `runs/` directory.

---

## 🗂 Notes

* All scripts are designed to be run from the `scripts/` directory.
* Ensure that `--data_dir` and `--data_matrices` paths remain consistent across all scripts.

---
