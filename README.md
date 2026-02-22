https://github.com/240IQCLIPS/IGL-Nav/releases
[![Release](https://img.shields.io/badge/IGL-Nav-Release-blue?style=for-the-badge)](https://github.com/240IQCLIPS/IGL-Nav/releases)

# IGL-Nav: Incremental 3D Gaussian Localization for Navigation

IGL-Nav is a research-oriented framework for incremental localization in 3D space, tailored for image-goal navigation. It leverages a probabilistic Gaussian representation to model the robot's pose and its local environment, updating beliefs as new image observations arrive. The approach emphasizes incremental refinement, efficiency, and robustness in dynamic or partially observed scenes. This README explains the goals, design choices, usage patterns, and practical details to help researchers and engineers reproduce experiments, extend the codebase, and apply IGL-Nav to related tasks.

---

## Table of contents

- Overview
- Core ideas and design principles
- Features at a glance
- System requirements
- Quick start
- Installation guide
- Data and experiments
- How IGL-Nav works
  - State representation
  - Observation model
  - Update rules
  - Localization loop
- Architecture and code structure
- Configuration and experimentation
- Evaluation and benchmarks
- Reproducibility and CI
- Datasets and benchmarks
- Visualization and debugging
- Extensions and interoperability
- Performance tips and tricks
- Troubleshooting
- Roadmap
- License and attribution
- Contributing

---

## Overview

IGL-Nav targets image-goal navigation scenarios where a mobile agent must reach a goal defined by a reference image, not by a fixed pose. The core idea is to maintain a probabilistic belief over the agent’s 3D pose and the nearby scene. This belief is represented as a collection of Gaussian components in 3D space, each encoding a potential pose hypothesis with a mean, covariance, and a weight. As the agent moves and acquires new images, the belief is updated in a principled manner, yielding an incremental, robust localization process that can guide navigation decisions.

This repository collects the ideas in a unified framework, including simulation-ready components, data pipelines, and experiment templates. The release assets and code aim to be accessible to researchers who want to validate results, reproduce experiments, or adapt the approach to related goals such as exploration, multi-robot coordination, or task-aware perception.

Note: The Releases page contains downloadable artifacts appropriate for different platforms. To obtain the latest stable build, you should visit the project’s Releases section and download the corresponding asset for your environment. The repository emphasizes reproducibility, making it straightforward to run the same experiments described in the accompanying papers and experiments.

- Primary motivation: provide a clean, modular path from perception to localization to navigation in image-goal tasks.
- Core strength: incremental updates to a 3D Gaussian belief, enabling stable performance in cluttered or partially observed scenes.
- Target users: researchers, robotics engineers, and ML practitioners exploring localization, perception, and navigation in 3D.

[Image: Illustration of incremental 3D Gaussian localization in a navigation task]
  - Note: This is a schematic visualization. In practice, the code uses a probabilistic representation of pose and a probabilistic model of observations to update beliefs.

---

## Core ideas and design principles

- Incrementality: Build the belief gradually as data accumulates, avoiding full re-computation from scratch.
- Gaussian representation: Use a set of 3D Gaussians to model the agent’s pose uncertainty and local structure.
- Image-goal guidance: The hallucination-free objective is to reach a region in the state space that matches a given image goal.
- Robustness: The method remains stable in noisy settings, with outliers handled via Bayesian weighting and resampling.
- Efficiency: The implementation favors vectorized computation, lightweight data structures, and practical hyperparameters to enable real-time or near real-time operation.
- Modularity: Clear separation between perception, probabilistic filtering, and navigation policy, allowing researchers to swap modules.

---

## Features at a glance

- Incremental 3D Gaussian state representation for robust localization
- Image-goal navigation support with goal-conditioned belief updates
- Probabilistic observation model that fuses visual input with 3D geometry
- Lightweight, modular codebase suitable for research experiments
- Reproducible experiments with configuration-driven pipelines
- Visualization utilities to inspect localization beliefs and trajectories
- Compatibility with common simulation environments and datasets

---

## System requirements

- Operating system: Linux or macOS; Windows is supported via WSL or native builds where provided
- Python: 3.8–3.11 recommended
- Numerical libraries: NumPy, SciPy
- Deep learning framework: PyTorch (stable release, with CUDA support if you plan to run GPU-accelerated perception)
- Visualization: Matplotlib (optional for plots), optional interactive tools
- CUDA (optional for GPU acceleration)
- Disk space: a few hundred megabytes for code and assets; more for large-scale datasets and benchmarks
- RAM: 8 GB or more for practical experiments; 16 GB or more for heavier workloads

You should be comfortable with Python environments and package management. If you use conda, a clean environment helps avoid dependency conflicts.

---

## Quick start

- Prepare a Python environment and install dependencies.
- Download the latest release asset from the Releases page and run the installer.
- Run a minimal demo to confirm the setup works on your machine.
- Inspect real-time localization and navigation with quick-look visualizations.

Key commands (illustrative; adapt to your environment):

- Create a new environment
  - conda create -n igl-nav python=3.10
  - conda activate igl-nav

- Install core dependencies
  - pip install -r requirements.txt
  - pip install torch torchvision

- Run a quick sanity check
  - python -m iglnav.scripts.check_setup --quick

- Launch a minimal demo
  - python -m iglnav.demo --config configs/default_demo.yaml

- Visualize intermediate beliefs
  - python -m iglnav.visualize --config configs/vis.yaml

The exact commands may differ slightly depending on how you configure paths and environments. The important part is to verify that the environment is set up properly and that you can trigger a basic demonstration that exercises the perception, belief update, and navigation modules.

If you want to explore further, you can consult the detailed sections that describe the configuration files, experiment templates, and advanced usage.

---

## Installation guide

This project emphasizes a clean, reproducible workflow. The following steps outline a typical installation process. Adjust them to fit your platform and preferences.

1) Clone the repository
- git clone https://github.com/240IQCLIPS/IGL-Nav.git
- cd IGL-Nav

2) Create a Python environment
- Use conda or venv to create an isolated environment.
- Example with conda: conda create -n igl-nav python=3.10; conda activate igl-nav

3) Install dependencies
- pip install -r requirements.txt
- If you plan to use CUDA, ensure it matches your PyTorch version and GPU driver. Install a CUDA-enabled PyTorch build if needed:
  - pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

4) Install optional tools for visualization and debugging
- pip install matplotlib seaborn

5) Verify the installation
- Run a small test so you confirm basic functionality (for example, a quick setup test script).

6) Prepare data paths
- Create data directories for datasets, logs, and results.
- Update paths in the config files to reflect your local directory structure.

7) Run a sample experiment
- Use a provided configuration file (e.g., configs/default_demo.yaml) and start the demo:
  - python -m iglnav.demo --config configs/default_demo.yaml

Notes:
- The repository uses configuration files to control experiment parameters. You can start from a minimal default and then gradually adjust hyperparameters for your experiments.
- If you use GPUs, ensure that the CUDA toolkit is installed and that PyTorch detects the devices. You can verify by running a quick torch.cuda.is_available() check.

---

## Data and experiments

IGL-Nav supports both synthetic and real-world data for image-goal navigation experiments. The synthetic data can help you validate the core localization logic under controlled conditions, while real-world sequences test robustness and practicality.

- Synthetic environments: simple layouts with 3D obstacles, configurable lighting, and camera viewpoints. These environments help you quantify localization accuracy and the effect of parameter choices.
- Realistic datasets: real-world recordings with ground-truth navigation goals or image correspondences. The goal is to align perception with a stable 3D Gaussian belief, enabling reliable navigation decisions.

In both cases, you can generate data streams that mimic camera frames, depth cues (if you have depth sensors), and ground-truth poses. These data streams feed into the observation model, which updates the Gaussian components representing the agent’s belief.

Experiment templates include:
- Localization accuracy vs. time
- Belief distribution visualization
- Trajectory planning and execution
- Sensitivity analysis to noise and occlusion
- Ablation studies for components of the observation model and update rules

You can tailor experiments by editing configuration files that describe:
- Sensor models (image noise, camera intrinsics, frame rate)
- Motion models (odometry error, prior distributions)
- Observation likelihoods (matching scores, feature extractors)
- Gaussian update strategies (weights, resampling thresholds)
- Planner behavior (greedy, stochastic, or learned policies)

Reproducibility is supported by fixed seeds in the configuration and by logging key metrics and artifacts. Each experiment can generate a results folder with:
- Trajectory data
- Belief states at each step
- Visualizations
- Logs with hyperparameters and random seeds

---

## How IGL-Nav works

This section presents the high-level ideas behind the system. It is designed to give you intuition, not every line of code.

State representation
- The agent’s pose is modeled in 3D: position (x, y, z) and orientation (roll, pitch, yaw) or an equivalent parameterization.
- The belief about the state is represented as a set of Gaussian components, each with a mean pose, a covariance describing uncertainty, and a weight indicating its plausibility.
- The Gaussian components allow the model to capture multiple plausible hypotheses when observations are ambiguous or occluded.

Observation model
- Observations are images acquired by the agent’s camera (and, if available, depth data).
- Visual features extracted from images (e.g., CNN-based descriptors) are compared to a set of learned or precomputed templates or to a latent representation that correlates with the 3D pose.
- The comparison yields a likelihood for each Gaussian component, which updates its weight.

Update rules
- We combine the prior belief with a likelihood from the new observation to produce a posterior belief. In the Gaussian mixture sense, this often involves updating means, covariances, and weights in a principled way.
- The update adapts the localization belief incrementally as new frames arrive, favoring components that explain the observations well.
- If the belief becomes too diffuse or a component’s weight falls below a threshold, pruning or resampling may occur to keep the representation compact.

Localization loop
- At each step, the system predicts the state using a motion model (which can be derived from odometry or a simple diffusion prior).
- The observation update refines the belief in light of current image data.
- The navigation policy uses the current belief to select actions that move toward the image-goal region in the latent space or the real 3D space, depending on how the planner is defined.
- The loop continues as long as the agent explores the environment, gathers observations, and refines its estimate of its own pose and the surrounding scene.

Key properties
- Incremental updates preserve stability in the presence of noise and partial observability.
- The Gaussian mixture cap ensures a balance between expressive power and computational efficiency.
- The method can operate in real time on mid-range hardware with sensible parameter choices.

Practical considerations
- The quality of the image-goal cue affects the convergence rate. Strong, distinctive goals help the system converge faster.
- The motion model accuracy influences how quickly the belief converges. Better odometry reduces the burden on the observation model.
- Proper tuning of weights, thresholds, and resampling strategies is essential for robust performance.

---

## Architecture and code structure

- iglnav/
  - __init__.py
  - core/
    - belief.py          # Gaussian belief management (means, covariances, weights)
    - motion_model.py    # Odometry and prior for prediction
    - observation.py       # Image-based likelihood and feature matching
    - update.py          # Bayesian update rules and resampling
  - models/
    - gaussian_kernel.py   # Core math for Gaussian components
    - feature_embeddings.py # Image feature extractor (optional)
  - planners/
    - navigator.py        # Action selection to move toward the image goal
    - policy.py           # Policy interfaces (greedy, learned, or hybrid)
  - utils/
    - data_utils.py       # Data loading and preprocessing
    - visualizers.py      # Belief visualization and plots
    - log_utils.py          # Logging and experiment tracking
  - configs/
    - default_demo.yaml
    - large_experiments.yaml
  - scripts/
    - check_setup.py
    - run_experiment.py
  - demo/
    - run_demo.py
  - docs/
    - architecture_diagram.png
    - README_more.pdf (optional)

Code is organized to separate concerns:
- Perception and observation models live in iglnav.core.observation.
- The probabilistic belief is managed by iglnav.core.belief.
- The motion model is a lightweight predictor in iglnav.core.motion_model.
- The planner and policy logic reside under iglnav.planners.
- Utilities for data I/O and visualization are in iglnav.utils.

The repository favors configuration-driven experiments, so most experiment parameters live in YAML files under iglnav.configs. You can copy a default file, rename it, and adjust the parameters to run different variants of the experiments. This design helps researchers compare different settings with minimal boilerplate.

---

## Configuration and experimentation

- Default configuration files provide sensible baselines for quick experiments.
- You can override parameters at runtime using command-line flags or by editing the YAML files directly.
- Parameters typically include:
  - Sensor model: camera intrinsics, image resolution, frame rate
  - Odometry model: drift characteristics, noise covariance
  - Observation model: feature extractor choice, similarity metric, likelihood temperature
  - Belief management: number of Gaussian components, resampling strategy, pruning thresholds
  - Planner: path to the image-goal curve in latent space or in 3D space, planning horizon
  - Logging: log intervals, tensorboard or CSV outputs, visualization toggles

Experiment templates cover common scenarios:
- Baseline localization accuracy under varying occlusion levels
- Sensitivity of the Gaussian component count
- Impact of observation noise on the convergence rate
- Real-time performance with different hardware profiles

When you run an experiment, the system logs:
- Per-step beliefs (means, covariances, weights)
- Pose estimates
- Observation features and likelihoods
- Actions and environment interactions
- Visualization outputs for debugging

---

## Evaluation and benchmarks

We recommend a careful evaluation to build intuition about IGL-Nav’s behavior:
- Localization accuracy: compare estimated pose against ground truth over time
- Convergence rate: time to reach a stable belief close to the true pose
- Robustness: behavior under partial observability and sensor noise
- Computational efficiency: time per update, memory usage

Standard metrics include:
- Root mean squared error (RMSE) of pose estimates
- Negative log-likelihood of observations given the belief
- Successful navigation rate within a tolerance of the goal
- Average number of Gaussian components needed to maintain acceptable accuracy
- Runtime per step and FPS for real-time applicability

Benchmarks can be run with synthetic data where ground truth is known precisely and with real-world data for practical relevance. Visualizations help interpret how the belief changes as the agent moves and as new observations arrive.

---

## Reproducibility and CI

- Reproducibility is a fundamental goal. Each run can be traced by seeds, configuration files, and logged artifacts.
- The project uses a CI workflow to validate:
  - Basic unit tests for belief operations
  - End-to-end sanity checks for a lightweight demo
  - Dependency checks to ensure consistent environments
- You can replicate results by:
  - Cloning the repository
  - Creating a clean environment
  - Running the provided demo with the same configuration and seed
  - Inspecting the resulting logs and figures

If you need to reproduce a figure or metric from a paper, you can replicate the exact steps by following the experiment templates and the controlled settings described in the docs.

---

## Datasets and benchmarks

IGL-Nav supports both synthetic and real-world datasets to illustrate the method's behavior in different conditions.

Synthetic data:
- Simple 3D layouts with controlled obstacles
- Adjustable lighting and camera noise
- Ground-truth trajectories for precise evaluation

Real-world data:
- Sequences captured from mobile robots or hand-held rigs
- Image-goal pairs derived from reference views
- Ground-truth poses for evaluation, when available

To extend or adapt, you can generate your own synthetic environments with configurable geometry and textures, or collect real-world data using a calibrated rig. Ensure that your dataset includes:
- Camera intrinsics (focal length, principal point)
- Time-stamped frames
- Optional depth information if available

---

## Visualization and debugging

Visualization is essential for understanding how the belief evolves and how the agent moves relative to its goal. The project includes utilities to visualize:
- The 3D Gaussian components and their weights
- The estimated pose trajectory over time
- The alignment between the current observation and the image-goal cue
- The navigation path taken by the agent

You can enable visualizations through configuration flags or run dedicated visualization scripts. Common outputs include:
- Trajectory plots
- Belief heatmaps in 2D slices of the 3D space
- Overlays of the current image with a reference image
- Per-step logs that you can review in a notebook or console

Illustrative visual content (where available) can help interpret where the model is confident and where ambiguity remains. Visual debugging proves especially helpful when adapting the model to new scenes.

---

## Extensions and interoperability

- Compatibility: The core concepts can interoperate with other perception pipelines, as the observation module is designed to accept a wide range of feature representations and similarity measures.
- Modularity: You can replace the feature extractor with your own model, swap the motion model with a different odometry source, or adopt a different planner without rewriting the entire codebase.
- Integrations: The framework is designed to plug into common robotics stacks, ROS-based workflows, or research environments with Python-based experimentation.

Future directions include:
- Multi-agent localization with shared Gaussian beliefs
- Learned proposal distributions for Gaussian components
- Active exploration strategies that balance information gain and goal progress
- Domain adaptation for image-goal cues across environments

---

## Performance tips and tricks

- Start with a small number of Gaussian components and gradually increase if the environment is highly ambiguous.
- Tune the resampling threshold to balance drift and computational load.
- Use a stable image feature extractor or precompute features for reference views to speed up likelihood computations.
- If you have GPU resources, enable GPU-accelerated parts of the observation model for faster feature matching and likelihood evaluation.
- Keep motion model noise conservative to avoid overconfidence that could trap the belief in an incorrect mode.
- Use visualization early in development to spot incorrect belief updates and to calibrate hyperparameters.

---

## Troubleshooting

- Import errors: Ensure you installed all dependencies from requirements.txt and that your Python environment is active.
- CUDA not found: Verify CUDA installation and ensure PyTorch matches your CUDA version.
- Slow performance: Check the number of Gaussian components; reduce them or enable pruning. Profile the observation module to identify bottlenecks.
- Bad convergence: Inspect the image-goal representation and ensure the goal is visually distinguishable in the dataset. Consider enriching the feature representation or adjusting the likelihood model.
- Data path issues: Confirm that data directories exist and that configuration paths point to the correct locations.

---

## Roadmap

- Enhance the observation model with richer multi-view cues to improve robustness in textureless regions.
- Introduce learned priors for the 3D Gaussian components to speed up convergence.
- Expand the visualization toolkit to support real-time interactive debugging.
- Extend support to additional sensors (e.g., depth or LiDAR) and fuse their information with the Gaussian beliefs.
- Improve sample efficiency with adaptive component management and better initialization strategies.

---

## License and attribution

IGL-Nav is released under a permissive license suitable for research and development. See the LICENSE file for details and attribution guidelines. If you use the code or components in your work, please cite the project or the corresponding papers and provide appropriate acknowledgments.

- BibTeX entries and citations are available in the docs or reference sections.
- Attribution for external resources, libraries, and datasets follows standard licenses as documented in the repository.

---

## Contributing

If you would like to contribute to IGL-Nav, please follow the guidelines in the CONTRIBUTING.md file. Contributions are welcome in the form of:
- Bug fixes
- New experiments or datasets
- Documentation improvements
- Novel observation models or planning strategies
- Tutorials and example notebooks

You can open issues to discuss ideas, propose enhancements, or report bugs. For code contributions, please create a feature branch, write tests, and submit a pull request with a clear description of the changes.

---

## Visual previews and references

- Gaussian representations in 3D space form the core of the belief model. This visual illustrates how a set of Gaussians can encode pose uncertainty and local geometry in a navigational scenario.
  - Figure: Gaussian components overlaid on a 3D scene to reflect the agent's belief at a given time.
  - Source: Conceptual diagrams for probabilistic localization and 3D Gaussian mixtures.

- A schematic of the image-goal navigation loop highlights how perception, belief updating, and planning interact to produce motion toward a target image view.

- For a general sense of 3D navigation visualization, see open-source resources that illustrate pose estimation, occupancy grids, and probability maps. The idea is to give a mental model of how beliefs evolve as the agent explores.

- Bonus figure: A baseline demonstration showing how the system updates the belief when observations are ambiguous (e.g., repetitive textures or occlusions). The figure helps explain the importance of incremental updates and the role of motion priors.

[Optional image gallery]
- Gaussian distribution
  - ![Gaussian distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Normal_Distribution_PDF.svg/640px-Normal_Distribution_PDF.svg.png)
- 3D navigation concept
  - ![IGL-Nav concept](https://picsum.photos/1200/420)

These figures serve as illustrative supports. They can be replaced with project-specific diagrams when you prepare official slides or a more formal manuscript.

---

## Release page and access

The project maintains a Releases page with downloadable artifacts for different platforms. The latest stable build is published there, and you can download the asset that matches your environment and run its installer or executable. The Releases page is the primary source for obtaining ready-to-run versions of IGL-Nav. For convenience, the link is displayed at the top of this page and a concise badge links to the same resource for quick access.

If you want to revisit the latest artifacts, navigate to the Releases section and select the artifact that corresponds to your setup. The release asset includes the code, prebuilt components, and scripts needed to reproduce the experiments described in the accompanying papers and documentation. The process is designed to be straightforward: download the asset, unpack it, and run the provided launcher or installer to get up and running quickly.

---

If you want to explore more, you can browse the documentation, experiment templates, and example configurations. The project is designed to be a living research resource, with ongoing enhancements and community contributions driving new capabilities.