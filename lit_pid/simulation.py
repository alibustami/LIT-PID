"""This module contains the function to run a single PyBullet simulation."""

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from lit_pid.pid_controller import PID


def run_simulation(
    robotId, gains, setpoint, total_time, dt, display_plot=False  # noqa
):
    """Run a single PyBullet simulation for a given set of PID gains. Calculates and returns performance metrics and the full angle time series."""
    # Reset robot to starting position
    startPos = [0, 0, 0.06]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(robotId, startPos, startOrientation)
    p.resetBaseVelocity(robotId, [0, 0, 0], [0, 0, 0])

    # Get joint indices
    num_joints = p.getNumJoints(robotId)
    joint_info = [p.getJointInfo(robotId, i) for i in range(num_joints)]
    joint_names = {info[1].decode("UTF-8"): info[0] for info in joint_info}
    left_rear_wheel_joint_idx = joint_names["left_rear_wheel_joint"]
    right_rear_wheel_joint_idx = joint_names["right_rear_wheel_joint"]

    pid = PID(Kp=gains["kp"], Ki=gains["ki"], Kd=gains["kd"])

    thetas, times = [], []
    steps = int(total_time / dt)
    WHEEL_RADIUS = 0.05
    ROBOT_WIDTH = 0.32

    for step in range(steps):
        t = step * dt
        pos, ori_quat = p.getBasePositionAndOrientation(robotId)
        current_theta_deg = np.degrees(p.getEulerFromQuaternion(ori_quat)[2])
        robot_angular_vel_rad_s = np.radians(
            pid.compute(setpoint, current_theta_deg, dt)
        )
        target_wheel_velocity = (robot_angular_vel_rad_s * ROBOT_WIDTH) / (
            2 * WHEEL_RADIUS
        )

        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=right_rear_wheel_joint_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_wheel_velocity,
            force=20,
        )
        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=left_rear_wheel_joint_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-target_wheel_velocity,
            force=20,
        )
        p.stepSimulation()
        thetas.append(current_theta_deg)
        times.append(t)

    # --- Performance Metric Calculations ---
    times_np = np.array(times)
    thetas_np = np.array(thetas)

    metrics = {
        "overshoot": float("inf"),
        "rise_time": float("inf"),
        "settling_time": float("inf"),
        "max_angle": float("-inf"),
    }

    max_value = np.max(thetas_np)
    metrics["max_angle"] = float(max_value)
    metrics["overshoot"] = (
        ((max_value - setpoint) / setpoint) * 100
        if max_value > setpoint
        else np.inf
    )

    try:
        t10_indices = np.where(thetas_np >= 0.1 * setpoint)[0]
        t90_indices = np.where(thetas_np >= 0.9 * setpoint)[0]
        if t10_indices.size > 0 and t90_indices.size > 0:
            t10 = times_np[t10_indices[0]]
            t90 = times_np[t90_indices[0]]
            metrics["rise_time"] = t90 - t10
    except IndexError:
        pass

    settling_band_upper = setpoint * 1.02
    settling_band_lower = setpoint * 0.98
    outside_band_indices = np.where(
        (thetas_np > settling_band_upper) | (thetas_np < settling_band_lower)
    )[0]

    if "rise_time" in metrics and metrics["rise_time"] != float("inf"):
        t90_index = t90_indices[0]
        final_outside_indices = outside_band_indices[
            outside_band_indices > t90_index
        ]
        if final_outside_indices.size > 0:
            last_outside_index = final_outside_indices[-1]
            if last_outside_index + 1 < len(times_np):
                metrics["settling_time"] = times_np[last_outside_index + 1]
        else:
            metrics["settling_time"] = times_np[t90_index]

    if display_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(
            times,
            thetas,
            "b-",
            label=f'Robot Heading (Kp={gains["kp"]}, Ki={gains["ki"]}, Kd={gains["kd"]})',
        )
        plt.axhline(
            y=setpoint,
            color="r",
            linestyle="--",
            label=f"Setpoint ({setpoint}°)",
        )
        plt.axhline(
            y=settling_band_upper,
            color="g",
            linestyle=":",
            label="2% Settling Band",
        )
        plt.axhline(y=settling_band_lower, color="g", linestyle=":")
        plt.title("Best PID Controller Performance", fontsize=16)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("Heading [degrees]", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("best_pid_performance.png")

    return metrics, times_np, thetas_np
