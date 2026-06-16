import numpy as np

from experiment import find_optimum
from utils import (
    get_activation,
    get_all_nonempty_activations,
    group_by_equality,
    merge_neurons,
)
import wandb
import time
import argparse

import matplotlib.pyplot as plt

# Replace with your project details
WANDB_ENTITY = ""  # Replace with your WandB entity/username
WANDB_PROJECT = ""  # Replace with your WandB project name

seed = np.random.randint(0, np.iinfo(np.int32).max)
np.random.seed(seed)


def get_xhat(d, perturbe=1e-3):
    xhat1 = np.zeros(d)
    xhat1[0] = 1
    xhat2 = np.zeros(d)
    xhat2[0] = 8 / 9
    xhat2[1] = -4 / 9
    xhat2[2] = 1 / 9
    xhat3 = np.zeros(d)
    xhat3[0] = 8 / 9
    xhat3[1] = 4 / 9
    xhat3[2] = 1 / 9
    xhats = [xhat1, xhat2, xhat3]
    for i in range(3, d):
        xhat = np.zeros(d)
        xhat[0] = 8 / 9
        xhat[i] = np.sqrt(17) / 9
        xhats.append(xhat)
    # combine in numpy array
    xhats = np.array(xhats)
    if perturbe:
        xhats += np.random.normal(0, perturbe, xhats.shape)

    # normalize
    xhats /= np.linalg.norm(xhats, axis=1, keepdims=True)
    return xhats


def get_teacher(d):
    teacher = np.zeros(d)
    teacher[0] = 4 / 5
    teacher[2] = 3 / 5
    return teacher


def get_initial_weights(d, width, alpha, symmetric_init=False):
    # Initialize the first layer weights with standard Gaussian scaled by alpha
    W = np.random.normal(0, alpha, (width, d))

    # Initialize the second layer weights with random sign and norm of corresponding first layer neuron
    a = np.zeros((1, width))
    for j in range(width):
        sign = 1 if np.random.rand() > 0.5 else -1
        norm = np.linalg.norm(W[j])
        a[0, j] = sign * norm

    if symmetric_init:
        W[width // 2 :, :] = W[: width // 2, :].copy()
        a[0, width // 2 :] = -a[0, : width // 2].copy()

    return W, a


def plotting(data, **scatter_kwargs):
    # project the data onto the plane
    data_proj = data @ plane
    # maintain direction of the vectors, but set their new length to the square root of their old length.
    data_proj = (
        data_proj
        / np.linalg.norm(data_proj, axis=1, keepdims=True)
        * np.power(np.linalg.norm(data_proj, axis=1, keepdims=True), 1 / 1)
    )
    plt.scatter(data_proj[:, 0], data_proj[:, 1], **scatter_kwargs)


def plot_neurons(W, a, **scatter_kwargs):
    W_pos_merged, W_neg_merged, _, _ = get_simplified_weights(W, a)
    plotting(W_pos_merged, marker="*", c="orange")
    plotting(W_neg_merged, marker="o", c="blue")


def get_simplified_weights(W, a):
    second_layer = a.squeeze()
    scaled_W = (W.T * np.abs(second_layer)).T
    W_pos = scaled_W[second_layer > 0]
    W_neg = scaled_W[second_layer < 0]
    merged_pos_neurons, max_pos_angle = merge_neurons(W_pos, zero_tolerance=1e-4)
    merged_neg_neurons, max_neg_angle = merge_neurons(W_neg, zero_tolerance=1e-4)
    return merged_pos_neurons, merged_neg_neurons, max_pos_angle, max_neg_angle


def train_network(
    X, y, d, alpha=0.001, lamb=0.1, lr=0.0001, width=4, num_epochs=1000, optimum=None
):
    W, a = get_initial_weights(d, width, alpha)

    reg_loss_history = []
    reg_loss_hisotry_length = 2
    log_growth = 1.01
    next_log_epoch = 0

    previous_activations = get_activation(X, W) * np.sign(a.T)
    activations_last_changed = 0

    single_neuron_activation_mismatch = False
    single_neuron_activation = None

    epoch = 0
    start_time = time.time() - 60
    while True:
        net_size = (W * W).sum() + (a * a).sum()
        first_layer_output = np.maximum(W @ X.T, 0)
        output = a @ first_layer_output
        residual = output - y
        grads = residual * first_layer_output * 2 / X.shape[0]
        second_layer_grads = np.sum(grads, axis=1, keepdims=True).T + 2 * lamb * a

        G = first_layer_output > 0
        first_layer_grads = (G * (2 * a.T @ residual)) @ X / X.shape[0] + 2 * lamb * W

        mse_loss = (residual**2).sum() / X.shape[0]
        loss = mse_loss + lamb * net_size

        W -= lr * first_layer_grads
        a -= lr * second_layer_grads

        terminate = (
            np.sum(np.square(first_layer_grads)) + np.sum(np.square(second_layer_grads))
            < 1e-16
        )

        if not np.allclose(previous_activations, get_activation(X, W) * np.sign(a.T)):
            previous_activations = get_activation(X, W) * np.sign(a.T)
            activations_last_changed = epoch

        if epoch >= next_log_epoch or terminate:
            acts = get_activation(X, W) * np.sign(a.T)
            acts = acts[np.linalg.norm(W * a.T, axis=1) > 1e-6]
            activation_groups = group_by_equality(W, acts)

            nuc = np.linalg.norm(W, ord="nuc")

            # compute effective rank
            # first compute svd of W, then consider the singular values of D. Then normalize the singular values by dividing each one with the sum of all values. Then compute the entropy of the normalized singular values, and take the exponential of that entropy to get the effective rank.
            D = np.linalg.svd(W, full_matrices=False, compute_uv=False)
            D_normalized = D / np.sum(D)
            effective_rank = np.exp(
                -np.sum(D_normalized * np.log(D_normalized + 1e-30))
            )

            reg_loss_history.append(loss)
            if len(reg_loss_history) > reg_loss_hisotry_length:
                reg_loss_history.pop(0)

            # Calculate the smallest and largest reg_loss values
            min_reg_loss = np.min(reg_loss_history)
            max_reg_loss = np.max(reg_loss_history)

            # take only those rows of W for which the corresponding element of a is positive
            W_pos = W[a.squeeze() > 0].copy()
            W_neg = W[a.squeeze() < 0].copy()

            W_pos /= np.linalg.norm(W_pos, axis=1, keepdims=True)
            W_neg /= np.linalg.norm(W_neg, axis=1, keepdims=True)

            pos_angles = W_pos @ W_pos.T
            neg_angles = W_neg @ W_neg.T

            max_pos_angle = np.min(np.abs(pos_angles))
            max_neg_angle = np.min(np.abs(neg_angles))

            pos_neurons, neg_neurons, max_pos_angle, max_neg_angle = (
                get_simplified_weights(W, a)
            )

            number_of_activation_regions = 0
            for key in activation_groups.keys():
                if np.all(activation_groups[key] == 0):
                    continue
                number_of_activation_regions += 1

            second_layer = a.squeeze()
            scaled_W = (W.T * np.abs(second_layer)).T
            W_pos = scaled_W[second_layer > 0]

            W_pos = W_pos[np.linalg.norm(W_pos, axis=1) > 1e-6]
            if len(W_pos) == 1:
                acts = get_activation(X, W_pos)
                if single_neuron_activation is None:
                    single_neuron_activation = acts
                else:
                    if single_neuron_activation != acts:
                        single_neuron_activation_mismatch = True
                        print("Single neuron activation mismatch")

            log_dict = {
                "epoch": epoch,
                "mse_loss": mse_loss,
                "net_size": net_size,
                "reg_loss": loss,
                "nuclear_norm": nuc,
                "effective_rank": effective_rank,
                "progress": (max_reg_loss - min_reg_loss) / min_reg_loss,
                "max_pos_angle": max_pos_angle,
                "max_neg_angle": max_neg_angle,
                "num_pos_neurons": len(pos_neurons),
                "num_neg_neurons": len(neg_neurons),
                "grads_square_norm": np.sum(np.square(first_layer_grads))
                + np.sum(np.square(second_layer_grads)),
                "pos_neurons": pos_neurons,
                "neg_neurons": neg_neurons,
                # "W": W.tolist(),
                # "a": a.tolist(),
                "activations_last_changed": activations_last_changed,
                "number_of_activation_regions": number_of_activation_regions,
                "single_neuron_activation_mismatch": single_neuron_activation_mismatch,
            }

            if optimum is not None:
                log_dict["reg_loss_distance"] = loss - optimum

            wandb.log(log_dict)

            if not terminate:
                next_log_epoch = epoch * log_growth

            elapsed_time = time.time() - start_time
            if elapsed_time >= 600 or terminate:  # 60 seconds = 1 minute
                start_time = time.time()  # Reset the timer

                plotting(teacher.reshape(1, -1), c="red", label="Teacher")
                plotting(optimum_sol, c="green", label="Optimum")
                plot_neurons(
                    W,
                    a,
                    c="blue",
                    label="Neurons",
                )
                plt.title(f"Epoch {epoch}")
                wandb.log({"projection": wandb.Image(plt)}, commit=False)
                plt.close()

        if terminate:
            print("Converged after", epoch, "epochs")
            break
        epoch += 1
        if epoch >= num_epochs:
            print("Reached maximum epochs without meeting target loss.")
            break

    return W, a


def compute_optimum(X, y, lamb):
    nonempty_activations = get_all_nonempty_activations(X)
    return find_optimum(X, y, lamb, nonempty_activations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the experiment with specified parameters."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Scaling factor for weight initialization.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for training."
    )
    parser.add_argument(
        "--lamb", type=float, default=0.00001, help="Regularization parameter."
    )
    parser.add_argument(
        "--d", type=int, default=4, help="Dimensionality of the input data."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100000000,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Width of the hidden layer in the network.",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=WANDB_PROJECT, help="WandB project name."
    )
    args = parser.parse_args()

    d = args.d
    alpha = args.alpha
    lamb = args.lamb
    lr = args.lr
    num_epochs = args.num_epochs
    width = args.width

    X = get_xhat(d)
    teacher = get_teacher(d)
    y = X @ teacher

    print("X:", X)
    print("Teacher:", teacher)
    print("y:", y)

    optimum, optimum_sol = compute_optimum(X, y, lamb)
    optima = [{"beta": lamb, "value": optimum}]
    print("Optimum solution", optimum_sol)
    print("Optimum value", optimum)

    H = (X @ X.T) / d

    eigenvalues = np.linalg.eigvals(H)
    print("Eigenvalues of H:", eigenvalues)

    _, _, Vt = np.linalg.svd(optimum_sol, full_matrices=False)
    plane = Vt[:2].T  # Top-2 principal components

    with wandb.init(
        project=args.wandb_project,
        entity=WANDB_ENTITY,
        config={
            "d": d,
            "lr": lr,
            "alpha": alpha,
            "lamb": lamb,
            "width": width,
            "optima": optima,
            "optimal_sol": optimum_sol,
            "X": X,
            "teacher": teacher,
            "seed": seed,
        },
        name=f"d-{d}-alpha-{alpha}-lamb-{lamb}-lr-{lr}-wdith-{width}",
    ):
        model = train_network(
            X,
            y,
            d,
            alpha=alpha,
            lamb=lamb,
            lr=lr,
            width=width,
            num_epochs=num_epochs,
            optimum=optimum,
        )
