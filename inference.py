import os
import torch
import numpy as np
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


torch.cuda.empty_cache()

# change the following paths
MODEL_PATH = "/media/yuxin/DiskBox/models/GR00T-N1.5-3B"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
EMBODIMENT_TAG = "UCL_Test_Bot"

device = "cuda" if torch.cuda.is_available() else "cpu"



data_config = DATA_CONFIG_MAP["UCL_Test_Bot"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

# print out the policy model architecture
print(policy.model)



modality_config = policy.modality_config

print("\n\n Modality Config:")
print(modality_config.keys())
print("\nKeys and shapes:")
for key, value in modality_config.items():
    if isinstance(value, np.ndarray):
        print(f"key: {key}, shape: {value.shape}")
    else:
        print(f"key: {key}, value: {value}")


# # Create the dataset
# dataset = LeRobotSingleDataset(
#     dataset_path=DATASET_PATH,
#     modality_configs=modality_config,
#     video_backend="decord",
#     video_backend_kwargs=None,
#     transforms=None,  # We'll handle transforms separately through the policy
#     embodiment_tag=EMBODIMENT_TAG,
# )

# step_data = dataset[0]



UCL_Test_Bot_step_data = {
    "video.gripper_view": np.zeros((1, 256, 256, 3), dtype=np.uint8),
    "state.arm_joint_positions": np.random.randn(1, 7).astype(np.float32),
    "state.eef_position": np.random.randn(1, 3).astype(np.float32),
    "state.eef_rotation": np.random.randn(1, 3).astype(np.float32),
    "action.arm_joint_positions": np.random.randn(16, 7).astype(np.float32),
    "action.eef_position": np.random.randn(16, 3).astype(np.float32),
    "action.eef_rotation": np.random.randn(16, 3).astype(np.float32),
    "annotation.human.action.task_description": [
        "pick the pear from the counter and place it in the plate"
    ],
}

step_data = UCL_Test_Bot_step_data

print(step_data)


print("\n\n GR00T Input:")
for key, value in step_data.items():
    if isinstance(value, np.ndarray):
        print(f"key: {key}, shape: {value.shape}")
    else:
        print(key, value)


print("\n\n Getting action from policy...")
predicted_action = policy.get_action(step_data)
for key, value in predicted_action.items():
    print(key, value.shape)