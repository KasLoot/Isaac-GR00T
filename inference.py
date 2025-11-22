import os
import torch
import numpy as np
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def get_policy(model_path: str = "/home/yuxin/models/GR00T-N1.5-3B", embodiment_tag: str = "UCL_Test_Bot", device: str = "cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data_config = DATA_CONFIG_MAP[embodiment_tag]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    modality_config = policy.modality_config

    print("\n\n Modality Config:")
    print(modality_config.keys())
    print("\nKeys and shapes:")
    for key, value in modality_config.items():
        if isinstance(value, np.ndarray):
            print(f"key: {key}, shape: {value.shape}")
        else:
            print(f"key: {key}, value: {value}")

    return policy


def run_inference_to_server(image: np.ndarray, instruction: str, ee_pos: np.ndarray, ee_quat: np.ndarray, joint_pos: np.ndarray,
                            prev_action: dict, policy: Gr00tPolicy):
    torch.cuda.empty_cache()

    USE_DOCKER_PATHS = False  # set to True if running in docker, and
    # change the following paths
    if USE_DOCKER_PATHS:
        MODEL_PATH = "/workspace/checkpoints/GR00T-N1.5-3B"
    else:
        MODEL_PATH = "/home/yuxin/models/GR00T-N1.5-3B"

    # REPO_PATH is the path of the pip install gr00t repo and one level up
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
    EMBODIMENT_TAG = "UCL_Test_Bot"

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"Using device: {device}")


    # data_config = DATA_CONFIG_MAP["UCL_Test_Bot"]
    # modality_config = data_config.modality_config()
    # modality_transform = data_config.transform()

    # policy = Gr00tPolicy(
    #     model_path=MODEL_PATH,
    #     embodiment_tag=EMBODIMENT_TAG,
    #     modality_config=modality_config,
    #     modality_transform=modality_transform,
    #     device=device,
    # )

    # print out the policy model architecture
    # print(policy.model)






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
    batch_size = 1
    image = image.reshape((batch_size, image.shape[0], image.shape[1], image.shape[2])).astype(np.uint8)
    joint_pos = joint_pos.reshape((batch_size, joint_pos.shape[0])).astype(np.float32)
    ee_pos = ee_pos.reshape((batch_size, ee_pos.shape[0])).astype(np.float32)
    ee_quat = ee_quat.reshape((batch_size, ee_quat.shape[0])).astype(np.float32)
    prev_action = prev_action
    prev_joint_pos = prev_action.get("action.arm_joint_positions", np.zeros((16, 9), dtype=np.float32))
    prev_eef_position = prev_action.get("action.eef_position", np.zeros((16, 3), dtype=np.float32))
    prev_eef_rotation = prev_action.get("action.eef_rotation", np.zeros((16, 4), dtype=np.float32))

    UCL_Test_Bot_step_data = {
        "video.gripper_view": image,
        "state.arm_joint_positions": joint_pos,
        "state.eef_position": ee_pos,
        "state.eef_rotation": ee_quat,
        "action.arm_joint_positions": prev_joint_pos,
        "action.eef_position": prev_eef_position,
        "action.eef_rotation": prev_eef_rotation,
        "annotation.human.action.task_description": [instruction],
    }

    step_data = UCL_Test_Bot_step_data

    # print(step_data)


    # print("\n\n GR00T Input:")
    # for key, value in step_data.items():
    #     if isinstance(value, np.ndarray):
    #         print(f"key: {key}, shape: {value.shape}")
    #     else:
    #         print(key, value)


    # print("\n\n Getting action from policy...")
    predicted_action = policy.get_action(step_data)
    # for key, value in predicted_action.items():
    #     print(key, value.shape)

    return predicted_action




def run_inference():
    torch.cuda.empty_cache()

    USE_DOCKER_PATHS = False  # set to True if running in docker, and
    # change the following paths
    if USE_DOCKER_PATHS:
        MODEL_PATH = "/workspace/checkpoints/GR00T-N1.5-3B"
    else:
        MODEL_PATH = "/home/yuxin/models/GR00T-N1.5-3B"

    # REPO_PATH is the path of the pip install gr00t repo and one level up
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
    EMBODIMENT_TAG = "UCL_Test_Bot"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")


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

    batch_size = 1

    UCL_Test_Bot_step_data = {
        "video.gripper_view": np.zeros((batch_size, 256, 256, 3), dtype=np.uint8),
        "state.arm_joint_positions": np.random.randn(batch_size, 9).astype(np.float32),
        "state.eef_position": np.random.randn(batch_size, 3).astype(np.float32),
        "state.eef_rotation": np.random.randn(batch_size, 4).astype(np.float32),
        "action.arm_joint_positions": np.random.randn(16, 9).astype(np.float32),
        "action.eef_position": np.random.randn(16, 3).astype(np.float32),
        "action.eef_rotation": np.random.randn(16, 4).astype(np.float32),
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
    print(predicted_action)
    for key, value in predicted_action.items():
        print(key, value.shape)


if __name__ == "__main__":
    run_inference()