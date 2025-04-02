# Spectrum Sharing Simulator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![NVIDIA Sionna](https://img.shields.io/badge/NVIDIA-Sionna-76B900.svg)](https://github.com/NVlabs/sionna)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI-Gym-0081A5.svg)](https://github.com/openai/gym)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System and Dependencies](#system-and-dependencies)
- [Installation and Example Run](#installation-and-example-run)
- [Outputs](#outputs)
- [Known Limitations](#known-limitations)
- [Repo Structure](#repo-structure)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)

## Overview

This repo offers a simulator for research into dynamic spectrum access (DSA) and spectrum sharing. Using Sionna for ray-tracing, deterministic wireless conditions are generated. The configurable Sionna simualation is encapsulated as an OpenAI gym environment, enabling a consistent interface for developing and training machine learning models for power control and dynamic spectrum access.

![Repo concept](https://github.com/user-attachments/assets/d2310f48-8223-4735-8128-dc898fb5dce7)

![Picture1](https://github.com/user-attachments/assets/ae1df354-b8e3-4235-b666-9089b7b6c7d8)

## Features

- 5G compliant, deterministic wireless simulations with NVIDIA Sionna.
- Configurable from top-level Hydra configuration file. 
- OpenAI Gym compatible environment interfaces.
- Modular to enable different scenarios.
- Inbuilt plotting and visualisation.
- Configurable spectrum sharing scenarios
- Extensible for custom scenarios and algorithms

## System and Dependencies
- Versions using Sionna v0.19.0 have sensitive pip dependencies, found in requirements.txt. Upgrade to Sionna v1.X.X coming soon.
- Python 3.10 is advised.
- Ubuntu 22.04 is advised.
- Tested on NVIDIA A100, L40 and A40 with CUDA 12.4 and Driver Version: 550.XXX.XX. Known issues with later and earlier drivers due to OptiX clashing with required versions of Mitsuba and DrJit.

## Installation and Example Run

```bash
# Clone the repository
git clone https://github.com/oscardilley/spectrum_sharing.git
cd spectrum_sharing

# Install dependencies
python3.10 -m venv .venv # create a virtual environment
source .venv/bin/activate # activate the venv
pip install -r requirements.txt
python3 -m spectrum_sharing.main 
```

**Note:** modify configuration through Config/simulation.yaml.

The above block runs _main.py_ which triggers the training routine that takes the following steps:
1. Load the configuration and initialise the environment and agent.
2. Load existing or create new model and replay buffer.
3. Start a loop for the specified number of episodes to run.
4. Take the initial observation from the environment and let the agent determine its action.
5. Observe the 5 tuple of (observation, reward, terminated, truncated, info) from the latest action.
6. Store the experience in the replay buffer.
7. Trigger agent training if the replay buffer is sufficiently large.
8. Render the visualisations.
9. Store rewards.
10. Regularly store model and buffer on disc.
11. Monitor terminated and truncated for exit command.
12. Loop until episode is complete.
13. Loop until all episodes have completed.

**Note:** in this implementation, each timestep of each episode is assumed to be 1s long to simplify calculations for velocity, rates, etc. 

## Outputs

### User Tracking on Coverage Maps

Primary maps:


<img src="https://github.com/user-attachments/assets/16f78537-65d9-4997-a487-6979b00e22a1" width="400">
<img src="https://github.com/user-attachments/assets/fd28e97d-68f9-4bf2-ac6b-6bab3465ce31" width="400">


Sharing maps:


<img src="https://github.com/user-attachments/assets/40cc38df-09ca-432c-95fa-22f35f6b951a" width="400">
<img src="https://github.com/user-attachments/assets/88a8f14b-431d-4aa1-927f-39fd2d7cbfe3" width="400">

### User Performance 
<img src="https://github.com/user-attachments/assets/695db8f2-2bf5-4ed7-ad30-9df79b0887bd">

### Reward Tracking Across Episodes
<img src="https://github.com/user-attachments/assets/de6f7892-382f-492e-8065-de99da1c3949">

### Scheduling Insights
![Scheduler_TX_1_Time_6](https://github.com/user-attachments/assets/33b3042e-1bfc-47cb-8350-6c3b36d2fcd1)

## Known Limitations

## Repo Structure

![repo structure](https://github.com/user-attachments/assets/30b186e2-be65-478c-8b2e-af67808b4118)

```bash
├── LICENSE
├── README.md
├── logging
│   └── app.log
├── requirements.txt
├── setup.py
└── spectrum_sharing
    ├── Archive
    ├── Buffer
    │   └── buffer.pickle
    ├── DQN_agent.py
    ├── Models
    ├── RL_simulator.py
    ├── Scene
    ├── Simulations
    ├── TestModels
    ├── __init__.py
    ├── __main__.py
    ├── benchmark.py
    ├── benchmarks.sh
    ├── channel_simulator.py
    ├── Config
    │   └── simulation.yaml
    ├── image_to_video.py
    ├── logger.py
    ├── main.py
    ├── plotting.py
    ├── scenario_simulator.py
    └── utils.py
```
You could benefit from this repo as follows:
- Implement a new RL algorithm by replacing _DQN_agent.py_ in _main.py_ with another agent for the same observation and action space.
- Change the agent as above and the observation and action space by modifying/ replacing _DQN_agent.py_ and _RL_simulator.py_.
- Implement your own scheduler in _scenario_simulator.py_.
- Modify the system parameters in _Config/simulation.yaml_.
- Add a new mobility model in _utils.py_.
- Change the model rewards in _RL_simulator.py_.

## Documentation

Comprehensive documentation coming sometime soon...

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{spectrum_sharing,
  author = {Oscar Dilley},
  title = {Spectrum Sharing: A Dynamic Spectrum Access Research Framework},
  year = {2025},
  url = {https://github.com/oscardilley/spectrum_sharing}
}
```
