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
- [Repo Structure](#repo-structure)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)

## Overview

This repo offers a simulator for research into dynamic spectrum access (DSA) and spectrum sharing. Using Sionna for ray-tracing, deterministic wireless conditions are generated. The configurable Sionna simualation is encapsulated as an OpenAI gym environment, enabling a consistent interface for developing and training machine learning models for power control and dynamic spectrum access.
![Repo concept](https://github.com/user-attachments/assets/d2310f48-8223-4735-8128-dc898fb5dce7)

## Features

- 5G compliant, deterministic wireless simulations with NVIDIA Sionna.
- Configurable from top-level Hydra configuration file. 
- OpenAI Gym compatible environment interfaces.
- Modular to enable different scenarios.
- Inbuilt plotting and visualisation.
- Configurable spectrum sharing scenarios
- Extensible for custom scenarios and algorithms

## System and Dependencies
- Versions using Sionna v0.19.0 have sensitive pip dependencies, found in requirements.txt
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

The above block runs _main.py_ which 

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
    ├── conf
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
- Modify the system parameters in _conf/simulation.yaml_.
- Add a new mobility model in _utils.py_.
- Change the model rewards in _RL_simulator.py_.

## Documentation

Comprehensive documentation coming soon...

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
