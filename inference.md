# Inference Using Custom Robot and Datasets

## Quick Start

1. Run inference code with dummy data.

```bash
python inference.py
```

2. You should get terminal outputs:

```
 GR00T Input:
key: video.gripper_view, shape: (1, 256, 256, 3)
key: state.arm_joint_positions, shape: (1, 7)
key: state.eef_position, shape: (1, 3)
key: state.eef_rotation, shape: (1, 3)
key: action.arm_joint_positions, shape: (16, 7)
key: action.eef_position, shape: (16, 3)
key: action.eef_rotation, shape: (16, 3)
annotation.human.action.task_description ['pick the pear from the counter and place it in the plate']


 Getting action from policy...
action.arm_joint_positions (16, 7)
action.eef_position (16, 3)
action.eef_rotation (16, 3)
```

## Inputs and Outputs

### Inputs

The model takes in the current observation, states, past predicted actions, and task description as inputs.

- video.gripper_view, shape: (1, 256, 256, 3): (batch_size, height, width, channel)
- state.arm_joint_positions, shape: (1, 7): (batch_size, num_joints)
- state.eef_position, shape: (1, 3): (batch_size, cartisian_positions_xyz)
- state.eef_rotation, shape: (1, 3): (batch_size, cartisian_orientations_rpy)
- action.arm_joint_positions, shape: (16, 7): (num_time_steps_action_predicted, num_joints)
- action.eef_position, shape: (16, 3): (num_time_steps_action_predicted, cartisian_positions_xyz)
- action.eef_rotation, shape: (16, 3): (num_time_steps_action_predicted, cartisian_orientations_rpy)
- annotation.human.action.task_description: A list of length batch_size containing the task description strings.

### Outputs

The model outputs the predicted actions for the next 16 time steps.

- action.arm_joint_positions (16, 7): (num_time_steps_action_predicted, num_joints)
- action.eef_position (16, 3): (num_time_steps_action_predicted, cartisian_positions_xyz)
- action.eef_rotation (16, 3): (num_time_steps_action_predicted, cartisian_orientations_rpy)

## Using Docker Container

1. Build the image using the docker file

```
docker build -f thor.Dockerfile -t isaac-groot-thor .
```

2. Create a container from the docker image

```
docker run -it --gpus all \
  -v /home/yuxin/Isaac-GR00T:/workspace \
  --name groot-container \
  isaac-groot-thor
```

## How to Add Your Own Robot Configuration?

### Add Data Config

1. Add your data config class in `./gr00t/experiment/data_config.py`
2. Add your DATA_CONFIG_MAP
3. In the inference code, set `data_config = DATA_CONFIG_MAP["UCL_Test_Bot"]`

```python
class UCL_Test_BotDataConfig(BaseDataConfig): # This is the new data config class
    video_keys = [
        "video.gripper_view",
    ]
    state_keys = [
        "state.arm_joint_positions",
        "state.eef_position",
        "state.eef_rotation"
    ]
    action_keys = [
        "action.arm_joint_positions",
        "action.eef_position",
        "action.eef_rotation",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

DATA_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "agibot_genie1": AgibotGenie1DataConfig(),
    "UCL_Test_Bot": UCL_Test_BotDataConfig(), # This is the new Data config map
}
```

### Add Embodiment Tag

1. Navigate to `gr00t/data/embodiment_tags.py`
2. Add your Embodiment Tag (Use this tag in your inference code: `EMBODIMENT_TAG = "UCL_Test_Bot"`)
3. Add your Embodiment Tag Mapping (you should use a number `< 31`)

```python
class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    UCL_TEST_BOT = "UCL_Test_Bot" # This is the new tag
    """
    The UCL Test Bot dataset.
    """

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    # Must be < max_num_embodiments (default 32). Use 31 like NEW_EMBODIMENT.
    EmbodimentTag.UCL_TEST_BOT.value: 30, # This is the new tag mapping
}
```

### Update Metadata in the Model Checkpoint Folder

For our UCL_Test_Bot, add the following to the metadata.json file. This file can be found in the experiment_cfg folder in your checkpoint directory.

```json
    "UCL_Test_Bot": {
        "statistics": {
            "state": {
                "eef_position": {
                    "max": [
                        1.0,
                        1.0,
                        1.0
                    ],
                    "min": [
                        -1.0,
                        -1.0,
                        -1.0
                    ],
                    "mean": [
                        0.005432123012930393,
                        0.010234567890123456,
                        -0.0023456789012345678
                    ],
                    "std": [
                        0.5123456789012345,
                        0.4987654321098765,
                        0.5234567890123456
                    ],
                    "q01": [
                        -0.9800000190734863,
                        -0.9700000286102295,
                        -0.9900000095367432
                    ],
                    "q99": [
                        0.9900000095367432,
                        0.9800000190734863,
                        0.9950000047683716
                    ]
                },
                "eef_rotation": {
                    "max": [
                        3.141592653589793,
                        3.141592653589793,
                        3.141592653589793
                    ],
                    "min": [
                        -3.141592653589793,
                        -3.141592653589793,
                        -3.141592653589793
                    ],
                    "mean": [
                        0.0012345678901234567,
                        -0.0023456789012345678,
                        0.0009876543210987654
                    ],
                    "std": [
                        1.5707963267948966,
                        1.5607963267948966,
                        1.5807963267948966
                    ],
                    "q01": [
                        -3.1000000000000005,
                        -3.1200000000000006,
                        -3.1100000000000005
                    ],
                    "q99": [
                        3.1000000000000005,
                        3.1200000000000006,
                        3.1100000000000005
                    ]
                },
                "arm_joint_positions": {
                    "max": [
                        2.0,
                        1.5,
                        2.5,
                        -0.5,
                        2.0,
                        3.0,
                        2.5
                    ],
                    "min": [
                        -2.0,
                        -1.5,
                        -2.5,
                        -3.0,
                        -2.0,
                        0.0,
                        -2.5
                    ],
                    "mean": [
                        0.0,
                        0.1,
                        -0.1,
                        -1.5,
                        0.0,
                        1.5,
                        0.0
                    ],
                    "std": [
                        0.5,
                        0.6,
                        0.4,
                        0.7,
                        0.6,
                        0.5,
                        0.8
                    ],
                    "q01": [
                        -1.8,
                        -1.2,
                        -2.0,
                        -2.5,
                        -1.5,
                        0.5,
                        -2.0
                    ],
                    "q99": [
                        1.8,
                        1.3,
                        1.9,
                        -0.5,
                        1.5,
                        2.5,
                        2.0
                    ]
                }
            },
            "action": {
                "eef_position": {
                    "max": [
                        1.0,
                        1.0,
                        1.0
                    ],
                    "min": [
                        -1.0,
                        -1.0,
                        -1.0
                    ],
                    "mean": [
                        0.00456790123456789,
                        0.009876543209876543,
                        -0.0012345678901234567
                    ],
                    "std": [
                        0.5123456789012345,
                        0.4987654321098765,
                        0.5234567890123456
                    ],
                    "q01": [
                        -0.9800000190734863,
                        -0.9700000286102295,
                        -0.9900000095367432
                    ],
                    "q99": [
                        0.9900000095367432,
                        0.9800000190734863,
                        0.9950000047683716
                    ]
                },
                "eef_rotation": {
                    "max": [
                        3.141592653589793,
                        3.141592653589793,
                        3.141592653589793
                    ],
                    "min": [
                        -3.141592653589793,
                        -3.141592653589793,
                        -3.141592653589793
                    ],
                    "mean": [
                        0.0012345678901234567,
                        -0.0023456789012345678,
                        0.0009876543210987654
                    ],
                    "std": [
                        1.5707963267948966,
                        1.5607963267948966,
                        1.5807963267948966
                    ],
                    "q01": [
                        -3.1000000000000005,
                        -3.1200000000000006,
                        -3.1100000000000005
                    ],
                    "q99": [
                        3.1000000000000005,
                        3.1200000000000006,
                        3.1100000000000005
                    ]
                },
                "arm_joint_positions": {
                    "max": [
                        2.0,
                        1.5,
                        2.5,
                        -0.5,
                        2.0,
                        3.0,
                        2.5
                    ],
                    "min": [
                        -2.0,
                        -1.5,
                        -2.5,
                        -3.0,
                        -2.0,
                        0.0,
                        -2.5
                    ],
                    "mean": [
                        0.0,
                        0.1,
                        -0.1,
                        -1.5,
                        0.0,
                        1.5,
                        0.0
                    ],
                    "std": [
                        0.5,
                        0.6,
                        0.4,
                        0.7,
                        0.6,
                        0.5,
                        0.8
                    ],
                    "q01": [
                        -1.8,
                        -1.2,
                        -2.0,
                        -2.5,
                        -1.5,
                        0.5,
                        -2.0
                    ],
                    "q99": [
                        1.8,
                        1.3,
                        1.9,
                        -0.5,
                        1.5,
                        2.5,
                        2.0
                    ]
                }
            },
            "total_trajectory_length": 5000000,
            "num_trajectories": 10000
        },
        "modalities": {
            "video": {
                "gripper_view": {
                    "resolution": [
                        256,
                        256
                    ],
                    "channels": 3,
                    "fps": 30.0
                }
            },
            "state": {
                "eef_position": {
                    "absolute": true,
                    "rotation_type": null,
                    "shape": [
                        3
                    ],
                    "continuous": true
                },
                "eef_rotation": {
                    "absolute": true,
                    "rotation_type": "euler_angles_rpy",
                    "shape": [
                        3
                    ],
                    "continuous": true
                },
                "arm_joint_positions": {
                    "absolute": true,
                    "rotation_type": null,
                    "shape": [
                        7
                    ],
                    "continuous": true
                }
            },
            "action": {
                "eef_position": {
                    "absolute": true,
                    "rotation_type": null,
                    "shape": [
                        3
                    ],
                    "continuous": true
                },
                "eef_rotation": {
                    "absolute": true,
                    "rotation_type": "euler_angles_rpy",
                    "shape": [
                        3
                    ],
                    "continuous": true
                },
                "arm_joint_positions": {
                    "absolute": true,
                    "rotation_type": null,
                    "shape": [
                        7
                    ],
                    "continuous": true
                }
            },
            "annotation": {
                "language": [
                    "task_description"
                ]
            }
        },
        "embodiment_tag": "UCL_Test_Bot"
    }
```
