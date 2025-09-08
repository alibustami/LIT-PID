"""Plots comparison graphs for PID tuning results from multiple models."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration --- #

# The base directory containing all your model result subdirectories
BASE_RESULTS_DIR = "results"

# --- NEW: Specify known best gains for EACH model ---
# The keys (e.g., 'results-gpt-5') MUST match the subdirectory names
# in your base results folder.
MODELS_BEST_GAINS = {
    "results-gpt-5": {"kp": 8.0, "ki": 0.1, "kd": 2.0},
    "results-gpt-5-mini": {"kp": 8.6, "ki": 0.055, "kd": 2.48},
    "results-gpt-5-nano": {"kp": 8.0, "ki": 0.12, "kd": 2.0},
}
# The name of the pickle file inside each model's directory
PICKLE_FILE_NAME = "all_iterations_data.pkl"


# --- Plotting Style --- #

# Set a style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["savefig.dpi"] = 300


def load_data(filename):
    """Load the PID tuning data from a pickle file."""
    if not os.path.exists(filename):
        print(f"  - Error: The file '{filename}' was not found.")
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(
            f"  - Successfully loaded data for {os.path.basename(os.path.dirname(filename))}. Found {len(data)} iterations."
        )
        return data
    except Exception as e:
        print(f"  - An error occurred while loading the pickle file: {e}")
        return None


def process_data_into_dataframe(data):
    """
    Processes the raw list of dictionaries into a pandas DataFrame.

    Args:
        data (list): The list of tuning results.

    Returns:
        pandas.DataFrame: A DataFrame containing the metrics and gains for each iteration.
    """
    records = []
    for item in data:
        record = {
            "iteration": item["iteration"],
            "kp": item["gains"]["kp"],
            "ki": item["gains"]["ki"],
            "kd": item["gains"]["kd"],
            "overshoot": item["metrics"]["overshoot"],
            "rise_time": item["metrics"]["rise_time"],
            "settling_time": item["metrics"]["settling_time"],
            "max_angle": item["metrics"]["max_angle"],
        }
        records.append(record)
    return pd.DataFrame(records)


def find_iteration_by_known_gains(df, known_gains, model_name):
    """
    Finds the iteration that most closely matches the provided PID gains and
    provides detailed feedback on the match.

    Args:
        df (pandas.DataFrame): The DataFrame of metrics and gains.
        known_gains (dict): A dictionary with 'kp', 'ki', 'kd' keys.
        model_name (str): Name of the current model for logging.

    Returns:
        int: The iteration number of the closest match.
    """
    gains_df = df[["kp", "ki", "kd"]]
    target_gains = np.array(
        [known_gains["kp"], known_gains["ki"], known_gains["kd"]]
    )

    # Calculate Euclidean distance for each row
    distances = np.linalg.norm(gains_df.values - target_gains, axis=1)

    # Find the index and value of the minimum distance
    closest_index = np.argmin(distances)
    min_distance = distances[closest_index]

    best_iteration_row = df.iloc[closest_index]

    # --- ENHANCED FEEDBACK ---
    print(f"  - For {model_name}:")
    print(
        f"    - Target gains (Kp, Ki, Kd)  : ({known_gains['kp']:.3f}, {known_gains['ki']:.3f}, {known_gains['kd']:.3f})"
    )
    print(
        f"    - Closest match found        : Iteration {int(best_iteration_row['iteration'])}"
    )
    print(
        f"    - Gains of actual match (Kp, Ki, Kd): ({best_iteration_row['kp']:.3f}, {best_iteration_row['ki']:.3f}, {best_iteration_row['kd']:.3f})"
    )
    print(f"    - Euclidean distance to target: {min_distance:.4f}")

    # Add a warning if the match is not very close. A high distance means the specified
    # "best gains" might not have a good representative in the data file.
    if min_distance > 1.0:  # This threshold can be adjusted
        print(
            f"    - WARNING: The closest match is relatively far from the target gains."
        )
        print(
            f"      This may cause the 'best performance' plot to look incorrect."
        )
        print(
            f"      Consider checking the gains for '{model_name}' in MODELS_BEST_GAINS."
        )

    return int(best_iteration_row["iteration"])


def calculate_performance_score(df):
    """
    Calculates a performance score for each iteration where lower is better.

    Args:
        df (pandas.DataFrame): The DataFrame of metrics.

    Returns:
        pandas.DataFrame: The DataFrame with an added 'score' column.
    """
    df_norm = df.copy()
    metrics = ["overshoot", "rise_time", "settling_time"]
    for col in metrics:
        if (df[col].max() - df[col].min()) == 0:
            df_norm[col] = 0
        else:
            df_norm[col] = (df[col] - df[col].min()) / (
                df[col].max() - df[col].min()
            )

    weights = {"overshoot": 0.4, "settling_time": 0.5, "rise_time": 0.1}

    df["score"] = (
        weights["overshoot"] * df_norm["overshoot"]
        + weights["settling_time"] * df_norm["settling_time"]
        + weights["rise_time"] * df_norm["rise_time"]
    )
    return df


def plot_all_step_responses(all_models_data, output_dir):
    """
    Plots all step responses for all models side-by-side in a single figure.
    """
    num_models = len(all_models_data)
    fig, axes = plt.subplots(
        1, num_models, figsize=(8 * num_models, 7), sharey=True
    )
    if num_models == 1:
        axes = [axes]  # Make it iterable if only one model

    # Use the maximum number of iterations across all models for consistent coloring
    max_iterations = (
        max(len(p_data["raw_data"]) for p_data in all_models_data.values())
        if all_models_data
        else 1
    )
    norm = plt.Normalize(vmin=1, vmax=max_iterations)
    cmap = plt.get_cmap("viridis")

    for ax, (model_name, processed_data) in zip(axes, all_models_data.items()):
        data = processed_data["raw_data"]
        best_iteration_num = processed_data["best_iter"]

        best_item_data = None
        for item in data:
            if item["iteration"] == best_iteration_num:
                best_item_data = item
                continue
            color = cmap(norm(item["iteration"]))
            ax.plot(
                item["steps"],
                item["angles"],
                color=color,
                linewidth=1.2,
                alpha=0.6,
            )

        if best_item_data:
            ax.plot(
                best_item_data["steps"],
                best_item_data["angles"],
                color="crimson",
                linewidth=2.5,
                # label=f"Best Trial (Iter. {best_iteration_num})",
                label="Best Trial",
                zorder=10,
            )

        ax.set_title(model_name, fontsize=20, weight="bold")
        ax.set_xlabel("Time (s)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.legend(fontsize=12)

    axes[0].set_ylabel("Angle (°)")
    fig.suptitle(
        "Comparison of Step Responses Across Models",
        fontsize=24,
        weight="bold",
    )

    # Add a single colorbar for the entire figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # --- FIX: Manually position the colorbar to prevent overlap ---
    # First, apply tight_layout to the subplots, leaving space on the right (right=0.95)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Then, create a new, dedicated axes for the colorbar in the space we created.
    # The list is [left, bottom, width, height] in figure-relative coordinates.
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Iteration Number", weight="bold")

    filename = os.path.join(output_dir, "comparison_step_responses.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nSaved combined step response plot to '{filename}'")


def plot_all_best_iterations(all_models_data, output_dir):
    """
    Plots the best iteration for all models side-by-side in a single figure.
    """
    num_models = len(all_models_data)
    fig, axes = plt.subplots(
        1, num_models, figsize=(8 * num_models, 7), sharey=True
    )
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, p_data) in zip(axes, all_models_data.items()):
        best_data = next(
            (
                item
                for item in p_data["raw_data"]
                if item["iteration"] == p_data["best_iter"]
            ),
            None,
        )
        if not best_data:
            continue

        metrics, gains = best_data["metrics"], best_data["gains"]
        ax.plot(
            best_data["steps"],
            best_data["angles"],
            color="royalblue",
            linewidth=2.5,
        )
        ax.set_title(model_name, fontsize=20)
        ax.set_xlabel("Time (s)")

        info_text = (
            f"$K_p$: {gains['kp']:.2f}, $K_i$: {gains['ki']:.2f}, $K_d$: {gains['kd']:.2f}\n"
            f"Overshoot: {metrics['overshoot']:.1f}%, Settling: {metrics['settling_time']:.1f}s"
        )
        ax.text(
            0.95,
            0.95,
            info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    axes[0].set_ylabel("Angle (°)")
    fig.suptitle(
        "Comparison of Best PID Performance", fontsize=24, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = os.path.join(output_dir, "comparison_best_performance.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved combined best performance plot to '{filename}'")


def plot_all_gains_search_spaces_3d(all_models_data, output_dir):
    """
    Creates a side-by-side 3D scatter plot of the gain search space for all models.
    """
    num_models = len(all_models_data)
    fig = plt.figure(figsize=(9 * num_models, 8))

    for i, (model_name, p_data) in enumerate(all_models_data.items()):
        ax = fig.add_subplot(1, num_models, i + 1, projection="3d")
        df = p_data["df"]
        best_iteration_num = p_data["best_iter"]

        sc = ax.scatter(
            df["kp"],
            df["ki"],
            df["kd"],
            c=df["iteration"],
            cmap="viridis",
            s=35,
            alpha=0.7,
        )
        ax.scatter(
            df.iloc[0]["kp"],
            df.iloc[0]["ki"],
            df.iloc[0]["kd"],
            c="blue",
            s=150,
            marker="o",
            edgecolor="black",
            label="Start",
            depthshade=False,
        )
        best_row = df[df["iteration"] == best_iteration_num]
        ax.scatter(
            best_row["kp"],
            best_row["ki"],
            best_row["kd"],
            c="red",
            s=200,
            marker="*",
            edgecolor="black",
            label="Best",
            depthshade=False,
        )

        ax.set_xlabel("$K_p$")
        ax.set_ylabel("$K_i$")
        ax.set_zlabel("$K_d$")
        ax.set_title(model_name, fontsize=18)
        ax.legend()
        ax.view_init(elev=20.0, azim=-65)

    fig.suptitle(
        "Comparison of PID Gains Search Space", fontsize=24, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = os.path.join(output_dir, "comparison_gains_search_space_3d.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved combined 3D gains search space plot to '{filename}'")


def setup_mock_results_directory():
    """Creates a mock directory structure and data for testing."""
    print("Setting up a mock results directory for demonstration...")
    if not os.path.exists(BASE_RESULTS_DIR):
        os.makedirs(BASE_RESULTS_DIR)

    for model_name, gains in MODELS_BEST_GAINS.items():
        model_path = os.path.join(BASE_RESULTS_DIR, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # --- Generate mock data for this model ---
        def generate_mock_response(kp, ki, kd, steps):
            damping = (kp / 20.0) + (kd / 10.0)
            frequency = 2.0 - (kp / 10.0)
            offset = ki * 0.1
            damping = max(0.1, damping)
            angles = (
                100
                * (1 - np.exp(-damping * steps) * np.cos(frequency * steps))
                + offset
            )
            angles += np.random.normal(0, 0.5, len(steps))
            return angles

        mock_data = []
        num_iterations = 44
        num_steps = 500
        time_end = 250
        steps = np.linspace(0, time_end, num_steps)

        for i in range(1, num_iterations + 1):
            kp_i = (
                2.0
                + 8.0 * (1 - np.exp(-i / 20.0))
                + np.random.uniform(-0.5, 0.5)
            )
            ki_i = (
                0.05
                + 0.5 * (1 - np.exp(-i / 20.0))
                + np.random.uniform(-0.05, 0.05)
            )
            kd_i = (
                0.5
                + 4.0 * (1 - np.exp(-i / 20.0))
                + np.random.uniform(-0.2, 0.2)
            )

            if i == num_iterations - 5:
                kp_i, ki_i, kd_i = gains["kp"], gains["ki"], gains["kd"]

            angles = generate_mock_response(kp_i, ki_i, kd_i, steps)
            max_angle = np.max(angles)

            try:
                settling_mask = np.where(
                    np.abs(angles - np.mean(angles[-50:])) > 5
                )[0]
                settling_time = (
                    steps[settling_mask[-1]]
                    if len(settling_mask) > 0
                    else time_end
                )
            except IndexError:
                settling_time = time_end
            try:
                rise_mask = np.where(angles >= max_angle * 0.9)[0]
                rise_time = (
                    steps[rise_mask[0]] if len(rise_mask) > 0 else time_end
                )
            except IndexError:
                rise_time = time_end
            overshoot = (
                ((max_angle - np.mean(angles[-50:])) / np.mean(angles[-50:]))
                * 100
                if np.mean(angles[-50:]) != 0
                else 0
            )

            mock_data.append(
                {
                    "iteration": i,
                    "gains": {"kp": kp_i, "ki": ki_i, "kd": kd_i},
                    "angles": angles,
                    "steps": steps,
                    "metrics": {
                        "overshoot": max(0, overshoot),
                        "rise_time": rise_time,
                        "settling_time": settling_time,
                        "max_angle": max_angle,
                    },
                }
            )

        pickle_path = os.path.join(model_path, PICKLE_FILE_NAME)
        with open(pickle_path, "wb") as f:
            pickle.dump(mock_data, f)
    print("Mock setup complete.\n")


def main():
    """
    Main function to load data for all models and generate comparison plots.
    """
    if not os.path.exists(BASE_RESULTS_DIR):
        print(f"Base directory '{BASE_RESULTS_DIR}' not found.")
        setup_mock_results_directory()

    all_models_processed_data = {}

    print("--- Loading and Processing Data for All Models ---")
    for model_name, known_gains in MODELS_BEST_GAINS.items():
        model_dir = os.path.join(BASE_RESULTS_DIR, model_name)
        if not os.path.isdir(model_dir):
            print(
                f"\nWarning: Directory for model '{model_name}' not found. Skipping."
            )
            continue

        pickle_path = os.path.join(model_dir, PICKLE_FILE_NAME)
        raw_data = load_data(pickle_path)
        if raw_data is None:
            continue

        df = process_data_into_dataframe(raw_data)
        df = calculate_performance_score(df)
        best_iter = find_iteration_by_known_gains(df, known_gains, model_name)

        all_models_processed_data[model_name] = {
            "raw_data": raw_data,
            "df": df,
            "best_iter": best_iter,
        }

    if not all_models_processed_data:
        print("\nNo data was loaded successfully. Exiting.")
        return

    print("\n--- Generating Comparison Plots ---")

    # Generate combined plots
    plot_all_step_responses(all_models_processed_data, BASE_RESULTS_DIR)
    plot_all_best_iterations(all_models_processed_data, BASE_RESULTS_DIR)
    # plot_all_gains_search_spaces_3d(all_models_processed_data, BASE_RESULTS_DIR)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
