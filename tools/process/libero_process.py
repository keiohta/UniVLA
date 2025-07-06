import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from socket import gethostname

# import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if gethostname().startswith('aic'):
    data_dir_original = "/home/user_00005_25b505/shared-storage/group_3/shared/datasets/libero/modified_libero_rlds"
    data_dir_processed = "/home/user_00005_25b505/shared-storage/group_3/shared/datasets/libero/processed_data/libero_all"
else:
    data_dir_original = "/media/ota/A292603492600F5B/data/airoa/modified_libero_rlds"
    data_dir_processed = "/media/ota/A292603492600F5B/data/airoa/processed_data/libero_all"

# Load the dataset, specify the data directory
for sub_dataname in ['libero_10_no_noops', 'libero_goal_no_noops', 'libero_object_no_noops', 'libero_spatial_no_noops']:
    dataset_dirs = f"{data_dir_original}/{sub_dataname}/1.0.0"
    builder = tfds.builder_from_directory(dataset_dirs)

    ds_all_dict = builder.as_dataset(split="train")

    # Set the output directory for processed data
    base_output_dir = data_dir_processed
    os.makedirs(base_output_dir, exist_ok=True)

    count = 0

    # Process the dataset and save with tqdm progress bar for episodes
    for episode in tqdm(ds_all_dict, desc="Processing episodes", unit="episode"):
        # tensor to string

        # name = episode["episode_metadata"]["episode_id"].numpy().decode()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        # concat last two elements of file_path
        name = file_path.split("/")[-2] + "__" + file_path.split("/")[-1].split(".")[0] + "__" + str(count)

        episode_dir = os.path.join(base_output_dir, name)
        os.makedirs(episode_dir, exist_ok=True)

        # Create a subdirectory for images
        image_dir = os.path.join(episode_dir, 'images')
        os.makedirs(image_dir, exist_ok=True)

        gripper_image_dir = os.path.join(episode_dir, 'gripper_images')
        os.makedirs(gripper_image_dir, exist_ok=True)

        # Create a subdirectory for actions
        action_dir = os.path.join(episode_dir, 'actions')
        os.makedirs(action_dir, exist_ok=True)

        # Prepare to store images and text
        images = []
        gripper_images = []
        languages = []
        actions = []

        for i, step in tqdm(enumerate(episode["steps"]), desc=f"Processing episode {name}", total=len(episode["steps"]),
                            unit="step"):
            observation = step["observation"]
            action = step["action"]

            image = observation["image"]
            image = Image.fromarray(image.numpy())
            gripper_image = observation["wrist_image"]
            gripper_image = Image.fromarray(gripper_image.numpy())

            language = step["language_instruction"].numpy().decode()

            images.append(image)
            languages.append(language)
            actions.append(action)
            gripper_images.append(gripper_image)

        # Save images
        for i in range(len(images)):
            image = images[i]
            image.save(os.path.join(image_dir, f"{i}.jpg"))
            gripper_image = gripper_images[i]
            gripper_image.save(os.path.join(gripper_image_dir, f"{i}.jpg"))
            action = actions[i]
            np.save(os.path.join(action_dir, f"{i}.npy"), action.numpy())
            if i == 0:
                with open(os.path.join(episode_dir, "instruction.txt"), "w") as f:
                    f.write(languages[i])

        count += 1
