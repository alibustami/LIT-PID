"""Main script to run the iterative PID tuning process using LLMs and PyBullet simulation."""

import os
import pickle

import pybullet as p
import pybullet_data

from lit_pid.csv_logger import save_history_to_csv
from lit_pid.llm_interaction import get_llm_tuned_gains
from lit_pid.simulation import run_simulation
from lit_pid.urdf_creator import create_robot_urdf

MODEL_NAME = "gpt-5"
MODEL_ARGS = {}
NUM_ITERATIONS = 44

results_folder = "results-{}".format(MODEL_NAME)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)


def main():
    """Execute function to run the PID tuning iterations."""
    total_time = 300.0  # seconds
    dt = 1.0 / 120.0  # simulation timestep

    first_4_iterations = {
        0: {
            "kp": 2.0,
            "ki": 0.05,
            "kd": 0.5,
        },  # Low gains for slow, stable response
        1: {
            "kp": 8.0,
            "ki": 0.1,
            "kd": 2.0,
        },  # Moderate gains for balanced response
        2: {
            "kp": 1.0,
            "ki": 0.6,
            "kd": 8.0,
        },  # Aggressive gains for fast, potentially oscillatory response
        3: {
            "kp": 5.0,
            "ki": 0.8,
            "kd": 15.0,
        },  # High integral and derivative for strong correction, possible overshoot
    }
    tuning_history = []

    physicsClient = p.connect(p.GUI)  # noqa
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")  # noqa
    urdf_file = create_robot_urdf()
    robotId = p.loadURDF(urdf_file)

    for i in range(NUM_ITERATIONS):
        print(f"\n{'='*35} TUNING ITERATION {i+1}/{NUM_ITERATIONS} {'='*35}")

        if i in first_4_iterations:
            current_gains = first_4_iterations[i]
            print(
                f">>> Using predefined gains for iteration {i+1}: Kp={current_gains['kp']}, Ki={current_gains['ki']}, Kd={current_gains['kd']}"
            )
        else:
            current_gains = get_llm_tuned_gains(
                tuning_history,
                use_api=True,
                model_args=MODEL_ARGS,
                model_name=MODEL_NAME,
            )
        if current_gains == (-1, -1, -1):
            print("Skipping iteration due to LLM/API error.")
            continue
        metrics, all_times, all_thetas = run_simulation(
            robotId=robotId,
            gains=current_gains,
            setpoint=90.0,
            total_time=total_time,
            dt=dt,
            display_plot=True,
        )

        # Store all results from the iteration
        tuning_history.append({"gains": current_gains, "metrics": metrics})

        # Store iteration data for pickle file
        iteration_data = {
            "iteration": i + 1,
            "gains": current_gains,
            "angles": all_thetas.tolist(),
            "steps": all_times.tolist(),
            "metrics": metrics,
        }

        # Load existing pickle data or create new list
        pickle_filename = "all_iterations_data.pkl"
        try:
            with open(
                os.path.join(results_folder, pickle_filename), "rb"
            ) as f:
                all_iterations_data = pickle.load(f)
        except FileNotFoundError:
            all_iterations_data = []

        # Add current iteration data
        all_iterations_data.append(iteration_data)

        # Save updated data to pickle file
        with open(os.path.join(results_folder, pickle_filename), "wb") as f:
            pickle.dump(all_iterations_data, f)
        print(f"[+] Saved iteration {i+1} data to {pickle_filename}")

        print("\n--- Performance Metrics ---")
        print(
            f"  Gains:             Kp={current_gains['kp']:.2f}, Ki={current_gains['ki']:.2f}, Kd={current_gains['kd']:.2f}"
        )
        print(f"  Percent Overshoot: {metrics['overshoot']:.2f}%")
        print(f"  Rise Time:         {metrics['rise_time']:.2f} s")
        print(f"  Settling Time:     {metrics['settling_time']:.2f} s")
        print(f"  Max Angle:         {metrics['max_angle']:.2f} deg")
        print("-" * 27)

        # Save the history to CSV after each iteration
        save_history_to_csv(
            tuning_history,
            filename=os.path.join(results_folder, "tuning_results.csv"),
        )

    p.disconnect()

    print(f"\n{'='*38} TUNING COMPLETE {'='*38}\n")
    print("--- Final Results Summary ---")

    # Simple cost function to find the best result
    best_entry = min(
        tuning_history,
        key=lambda x: 2.0 * x["metrics"]["overshoot"]
        + x["metrics"]["rise_time"]
        + x["metrics"]["settling_time"],
    )
    best_gains = best_entry["gains"]
    best_metrics = best_entry["metrics"]

    print(
        f"Best performance achieved with Kp={best_gains['kp']}, Ki={best_gains['ki']}, Kd={best_gains['kd']}"
    )
    print(f"  - Overshoot: {best_metrics['overshoot']:.2f}%")
    print(f"  - Rise Time: {best_metrics['rise_time']:.2f}s")
    print(f"  - Settling Time: {best_metrics['settling_time']:.2f}s\n")

    print("Displaying plot for the best performing controller...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    robotId_final = p.loadURDF(urdf_file)

    run_simulation(
        robotId=robotId_final,
        gains=best_gains,
        setpoint=90.0,
        total_time=total_time,
        dt=dt,
        display_plot=True,
    )
    p.disconnect()

    if os.path.exists(urdf_file):
        os.remove(urdf_file)


if __name__ == "__main__":
    main()
