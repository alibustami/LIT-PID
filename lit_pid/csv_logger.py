"""Module for saving PID tuning history to a CSV file."""

import csv


def save_history_to_csv(history, filename):
    """Save the tuning history to a CSV file."""
    if not history:
        return

    headers = [
        "Iteration",
        "Kp",
        "Ki",
        "Kd",
        "Overshoot (%)",
        "Rise Time (s)",
        "Settling Time (s)",
        "Max Angle (deg)",
    ]

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for i, entry in enumerate(history):
                g = entry["gains"]
                m = entry["metrics"]
                row = [
                    i + 1,
                    f"{g['kp']:.3f}",
                    f"{g['ki']:.3f}",
                    f"{g['kd']:.3f}",
                    f"{m['overshoot']:.3f}",
                    f"{m['rise_time']:.3f}",
                    f"{m['settling_time']:.3f}",
                    f"{m.get('max_angle', float('nan')):.3f}",
                ]
                writer.writerow(row)
        print(f"\n[+] Successfully saved tuning history to {filename}")
    except IOError as e:
        print(f"\n[!] Error saving CSV file: {e}")
