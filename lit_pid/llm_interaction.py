"""Handles interaction with the OpenAI API to get new PID gains based on tuning history."""

import os
import re
import textwrap
from time import time

from openai import OpenAI


def get_llm_tuned_gains(history, model_name, model_args={}, use_api=True):
    """Generate a prompt, calls the OpenAI API, and returns new PID gains. Falls back to a simulated response if the API call fails."""
    iteration = len(history)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not use_api:
        if not api_key and use_api:
            print("\nWARNING: OPENAI_API_KEY environment variable not set.")
        return -1, -1, -1  # Indicate failure to use API

    client = OpenAI(api_key=api_key)

    prompt = """You are a control systems expert. Your task is to tune the PID gains (Kp, Ki, Kd) for a simulated robot's orientation control to reach a 90-degree setpoint.

We want to minimize three key performance metrics (lower is better):
1.  **Overshoot (%)**: How much the robot over-turns past the setpoint.
2.  **Rise Time (s)**: Time taken to go from 10% to 90% of the setpoint.
3.  **Settling Time (s)**: Time until the robot's angle stays within 2% of the setpoint.

Here is the history of previous tuning attempts:
"""
    if not history:
        prompt += "\nThis is the first iteration. A reasonable starting point for a system like this might be Kp=2.0, Ki=0.0, Kd=0.1."
    else:
        header = "| Iteration |    Kp |    Ki |    Kd | Overshoot (%) | Rise Time (s) | Settling Time (s) |"
        prompt += f"\n{header}\n"
        prompt += "|-----------|-------|-------|-------|---------------|---------------|-------------------|\n"
        for i, entry in enumerate(history):
            g = entry["gains"]
            m = entry["metrics"]
            prompt += (
                f"| {i+1:<10}| {g['kp']:<6.2f}| {g['ki']:<6.2f}| {g['kd']:<6.2f}| "
                f"{m['overshoot']:<14.2f}| {m['rise_time']:<14.2f}| {m['settling_time']:<18.2f}|\n"
            )

    prompt += "\nBased on this history, analyze the previous result and propose a new set of (Kp, Ki, Kd) values for the next iteration. Provide ONLY the three comma-separated numbers for Kp, Ki, and Kd and nothing else."
    prompt += "\nRespond in the format: ```{ Kp, Ki, Kd }```"
    print("=" * 80)
    print(f"--- PROMPT FOR LLM (ITERATION {iteration + 1}) ---")
    print(textwrap.dedent(prompt))
    print("=" * 80)

    try:
        start_time = time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant expert in control systems engineering. You will be given a history of PID tuning trials. Your task is to respond with the next best guess for Kp, Ki, and Kd values. Respond ONLY with three comma-separated numbers (e.g., '5.0, 0.2, 8.5'). Do not include any explanation, labels, or additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            **model_args,
        )
        end_time = time()
        print(
            f"\n>>> OpenAI API call completed in {end_time - start_time:.2f} seconds."
        )
        llm_output = response.choices[0].message.content.strip()
        print(f"\n>>> Raw LLM Response: '{llm_output}'")

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", llm_output)

        if len(numbers) == 3:
            kp, ki, kd = map(float, numbers)
            next_gains = {"kp": kp, "ki": ki, "kd": kd}
            print(
                f">>> Parsed Gains: Kp={next_gains['kp']}, Ki={next_gains['ki']}, Kd={next_gains['kd']}\n"
            )
            return next_gains
        else:
            print(
                f"WARNING: LLM response '{llm_output}' was not in the expected format (3 numbers)."
            )
            return -1, -1, -1

    except Exception as e:
        print(f"\nERROR: OpenAI API call failed: {e}")
        return -1, -1, -1
