from gr00t.utils.eval import calc_mse_for_single_trajectory
import warnings
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.simplefilter("ignore", category=FutureWarning)

USE_DOCKER_PATHS = True  # set to True if running in docker, and
if USE_DOCKER_PATHS:

    PRE_TRAINED_MODEL_PATH = "/workspace/checkpoints/GR00T-N1.5-3B"
    EMBODIMENT_TAG = EmbodimentTag.UCL_TEST_BOT
    DATASET_PATH = "/workspace/demo_data/robot_sim.PickNPlace"

else:

    PRE_TRAINED_MODEL_PATH = "/media/yuxin/DiskBox/models/GR00T-N1.5-3B"
    EMBODIMENT_TAG = EmbodimentTag.UCL_TEST_BOT
    DATASET_PATH = "../demo_data/robot_sim.PickNPlace"

data_config = DATA_CONFIG_MAP["UCL_Test_Bot"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()


pre_trained_policy = Gr00tPolicy(
    model_path=PRE_TRAINED_MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)


mse = calc_mse_for_single_trajectory(
    pre_trained_policy,
    dataset,
    traj_id=0,
    modality_keys=["right_arm", "right_hand"],   # we will only evaluate the right arm and right hand
    steps=150,
    action_horizon=16,
    plot=True
)

print("MSE loss for trajectory 0:", mse)
