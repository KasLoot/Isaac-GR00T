# Inference Using Custom Robot and Datasets
## Quick Start
Run inference code with dummy data.
```bash
python inference.py
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