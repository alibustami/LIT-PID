# LIT-PID

**LIT‑PID: Language‑in‑the‑Loop PID Auto‑Tuning for Differential‑Drive Mobile Robots**

## Overview

LIT-PID is a Python-based framework for automatic PID controller tuning using language models (LLMs) in the loop. It is designed for differential-drive mobile robots and leverages simulation, logging, and URDF-based robot modeling to facilitate research and experimentation in intelligent control systems.


![system architecture](figures/Lit%20PID%20Lucidchart.jpg)

## Features

- **Language-in-the-Loop PID Tuning:** Integrates LLMs to suggest and refine PID parameters based on simulation results.
- **Differential-Drive Robot Simulation:** Simulates four-wheel robots using URDF models and physics engines.
- **Automated Experiment Logging:** Records simulation data and tuning results for analysis and reproducibility.
- **Visualization:** Generates plots and figures to compare controller performance.
- **Extensible Architecture:** Modular codebase for easy adaptation to other robot types or control strategies.

## Directory Structure

```
lit_pid/
	__init__.py
	csv_logger.py         # Handles experiment data logging to CSV
	four_wheel_robot.urdf # URDF model for the robot
	llm_interaction.py    # Interfaces with LLMs for PID suggestions
	main.py               # Main entry point for running experiments
	pid_controller.py     # PID controller implementation
	simulation.py         # Robot simulation environment
	urdf_creator.py       # Utilities for URDF generation
figures/
	bad.png
	Lit PID Lucidchart.pdf
results/
	results-gpt-5/
	results-gpt-5-mini/
	results-gpt-5-nano/
	# Each contains experiment data and plots
```

## Getting Started

### Prerequisites

- Python 3.11+
- [pybullet](https://pybullet.org/) (for simulation)
- [uv](https://github.com/astral-sh/uv) (for fast Python package management)
- Other dependencies as listed in `pyproject.toml`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alibustami/LIT-PID.git
   cd LIT-PID
   ```

2. Create and activate a virtual environment:
   ```bash
   make create-venv
   ```


### Usage

Before running, export your OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here
```

Run the main experiment script:
```bash
uv run lit_pid/main.py
```

- Results and logs will be saved in the `results/` directory.
- Figures and diagrams are available in the `figures/` folder.

### Customization

- Modify `lit_pid/four_wheel_robot.urdf` to change robot parameters.
- Adjust simulation settings in `lit_pid/simulation.py`.
- Integrate with different LLMs via `lit_pid/llm_interaction.py`.

## Results

Experiment outputs are organized in the `results/` folder, with subfolders for different LLM configurations (e.g., `results-gpt-5`, `results-gpt-5-mini`, `results-gpt-5-nano`). Each contains:

- `all_iterations_data.pkl`: Pickled data for all tuning iterations.
- `tuning_results.csv`: CSV log of tuning results.
- `best-response.png`: Visualization of best controller response.

## Figures

- `figures/bad.png`: Example of poor controller performance.
- `figures/Lit PID Lucidchart.pdf`: System architecture diagram.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Inspired by research in intelligent control and robotics.
- Uses [pybullet](https://pybullet.org/) for simulation.
