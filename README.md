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
![Rewards_Tracker_20250527_122134](https://github.com/user-attachments/assets/bbf527fd-ce5d-4fe2-909f-f08bffd820e1)

### Scheduling Insights
![Scheduler_TX_tx1_Time_0](https://github.com/user-attachments/assets/cbfdb0a5-c2ae-4988-96cb-df62b16cf839)

## Known Limitations

The version of Sionna used is highly dependent on the Optix driver and does not work with NVIDIA 570 drivers. Tested and working with 550-server.

In Sionna 0.19.2/ solver_cm.py. Remove below as to patch intermittent error:  "CRITICAL - Coverage map generation failed: Attempt to convert a value (None) with an unsupported type (class 'NoneType') to a Tensor." linked to RIS. Note: only appropriate if not using RIS. Lines 2854->

```bash
  File "/home/ubuntu/spectrum_sharing_v0/spectrum_sharing/scenario_simulator.py", line 280, in __call__
    self.cm, self.sinr = self._coverage_map() 
  File "/home/ubuntu/spectrum_sharing_v0/spectrum_sharing/scenario_simulator.py", line 240, in _coverage_map
    raise e
  File "/home/ubuntu/spectrum_sharing_v0/spectrum_sharing/scenario_simulator.py", line 226, in _coverage_map
    cm = self.scene.coverage_map(max_depth=self.max_depth,           # Maximum number of ray scene interactions
  File "/home/ubuntu/.local/lib/python3.10/site-packages/sionna/rt/scene.py", line 1363, in coverage_map
    output = self._solver_cm(max_depth=max_depth,
  File "/home/ubuntu/.local/lib/python3.10/site-packages/sionna/rt/solver_cm.py", line 215, in __call__
    cm, los_primitives = self._shoot_and_bounce(meas_plane,
  File "/home/ubuntu/.local/lib/python3.10/site-packages/sionna/rt/solver_cm.py", line 2863, in _shoot_and_bounce
    ris_ang_opening = self._apply_ris_reflection(ris_reflect_ind,
  File "/home/ubuntu/.local/lib/python3.10/site-packages/sionna/rt/solver_cm.py", line 2347, in _apply_ris_reflection
    act_data = self._extract_active_ris_rays(active_ind, int_point,
  File "/home/ubuntu/.local/lib/python3.10/site-packages/sionna/rt/solver_cm.py", line 1663, in _extract_active_ris_rays
    act_radii_curv = tf.gather(radii_curv, active_ind, axis=0)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/ubuntu/.local/lib/python3.10/site-packages/tensorflow/python/framework/constant_op.py", line 103, in convert_to_eager_tensor
    return ops.EagerTensor(value, ctx.device_name, dtype)
ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.
```

```python
            if tf.shape(ris_reflect_ind)[0] > 0:
                # ris_e_field : [num_ris_reflected_samples, num_tx_patterns, 2]
                # ris_field_es : [num_ris_reflected_samples, 3]
                # ris_field_ep : [num_ris_reflected_samples, 3]
                # ris_int_point : [num_ris_reflected_samples, 3]
                # ris_k_r : [num_ris_reflected_samples, 3]
                # ris_n : [num_ris_reflected_samples, 3]
                # ris_radii_curv : [num_ris_reflected_samples, 2]
                # ris_dirs_curv : [num_ris_reflected_samples, 2, 3]
                # ris_ang_opening : [num_ris_reflected_samples]
                ris_e_field, ris_field_es, ris_field_ep, ris_int_point,\
                 ris_k_r, ris_n, ris_radii_curv, ris_dirs_curv,\
                 ris_ang_opening = self._apply_ris_reflection(ris_reflect_ind,
                        int_point, previous_int_point, primitives, e_field,
                        field_es, field_ep, radii_curv, dirs_curv,
                        angular_opening)
                updated_e_field = tf.concat([updated_e_field, ris_e_field],
                                            axis=0)
                updated_field_es = tf.concat([updated_field_es, ris_field_es],
                                                axis=0)
                updated_field_ep = tf.concat([updated_field_ep, ris_field_ep],
                                                axis=0)
                updated_int_point = tf.concat([updated_int_point,
                                                ris_int_point], axis=0)
                updated_k_r = tf.concat([updated_k_r, ris_k_r], axis=0)
                normals = tf.concat([normals, ris_n], axis=0)
                updated_radii_curv = tf.concat([updated_radii_curv,
                                                ris_radii_curv], axis=0)
                updated_dirs_curv = tf.concat([updated_dirs_curv,
                                            ris_dirs_curv], axis=0)
                updated_ang_opening = tf.concat([updated_ang_opening,
                                                ris_ang_opening], axis=0)
```

## Repo Structure

![repo structure](https://github.com/user-attachments/assets/30b186e2-be65-478c-8b2e-af67808b4118)

```bash
├── logging
│   └── app.log
├── spectrum_sharing
│   ├── Archive
│   ├── Assets
│   │   ├── Maps
│   │   ├── grid.npy
│   │   ├── primary_maps.npy
│   │   └── sharing_maps.npy
│   ├── Buffer
│   │   └── buffer.pickle
│   ├── Config
│   │   ├── preprocessing.yaml
│   │   └── simulation.yaml
│   ├── Models
│   │   ├── model
│   │   └── model_target
│   ├── Results
│   │   └── results_20250519_103931.csv
│   ├── Scene
│   │   ├── BRISTOL_3
│   │   │   ├── mesh
│   │   │   └── simple_OSM_scene.xml
│   │   ├── OSM_to_Sionna (4).ipynb
│   │   └── ground_plane.obj
│   ├── Simulations
│   │   ├── Plot for band 1, tx0.png
│   │   ├── Plot for band 1, tx1.png
│   │   ├── Rewards_Ep_0.png
│   │   ├── Rewards_Tracker.png
│   │   ├── Scheduler_TX_0_Time_0.png
│   │   ├── Scheduler_TX_1_Time_0.png
│   │   ├── cam1_primary_scene.png
│   │   ├── cam1_sharing_scene.png
│   │   ├── cam2_primary_scene.png
│   │   └── cam2_sharing_scene.png
│   ├── TestModels
│   │   ├── model
│   │   └── model_target
│   ├── Tests
│   ├── Videos
│   ├── DQN_agent.py
│   ├── RL_simulator.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── benchmarks.sh
│   ├── channel_simulator.py
│   ├── image_to_video.py
│   ├── logger.py
│   ├── main.py
│   ├── plotting.py
│   ├── preprocessing.py
│   ├── scenario_simulator.py
│   ├── utils.py
│   └── validate_preprocess.py
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
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
