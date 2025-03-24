# Spectrum Sharing

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA Sionna](https://img.shields.io/badge/NVIDIA-Sionna-76B900.svg)](https://github.com/NVlabs/sionna)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI-Gym-0081A5.svg)](https://github.com/openai/gym)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

A dynamic spectrum access research framework that combines NVIDIA Sionna ray tracing with OpenAI Gym compatible environments for training reinforcement learning agents.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

Spectrum_Sharing creates realistic wireless communications environments using NVIDIA Sionna's ray tracing capabilities and wraps them in OpenAI Gym compatible interfaces. This enables researchers to train and evaluate reinforcement learning agents for dynamic spectrum access scenarios.

## Features

- Realistic RF propagation modeling with NVIDIA Sionna ray tracing
- OpenAI Gym compatible environment interfaces
- Configurable spectrum sharing scenarios
- Built-in reinforcement learning agent implementations
- Visualization tools for environment and agent performance
- Extensible framework for custom scenarios and algorithms

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spectrum_sharing.git
cd spectrum_sharing

# Install dependencies
pip install -e .
```

## Getting Started

```python
import gymnasium as gym
import spectrum_sharing

# Create a spectrum sharing environment
env = gym.make('spectrum-sharing-v0')

# Reset the environment
observation, info = env.reset()

# Interact with the environment
for _ in range(1000):
    action = env.action_space.sample()  # Your agent's policy here
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
```

## Examples

See the [examples](./examples) directory for detailed usage examples:

- Basic environment interaction
- Training a DQN agent
- Custom scenario configuration
- Visualization and analysis

## Documentation

Comprehensive documentation is available at [docs/index.md](docs/index.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{spectrum_sharing,
  author = {Your Name},
  title = {Spectrum Sharing: A Dynamic Spectrum Access Research Framework},
  year = {2025},
  url = {https://github.com/yourusername/spectrum_sharing}
}
```
