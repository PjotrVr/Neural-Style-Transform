import os
import subprocess
import shutil

import cv2 as cv
import numpy as np    
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from models.vgg_nets import Vgg16, Vgg19


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(image_path, target_shape=None):
    if not os.path.exists(image_path):
        raise Exception(f"Path does not exist: {image_path}")

    image = cv.imread(image_path)[:, :, ::-1]

    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = image.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        else:
            image = cv.resize(image, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    image = image.astype(np.float32)
    image /= 255.0  # Pixel values range [0, 1]
    return image


def prepare_image(image_path, target_shape, device):
    image = load_image(image_path, target_shape=target_shape)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    image = transform(image).to(device).unsqueeze(0)

    return image


def save_image(image, image_path):
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    cv.imwrite(image_path, image[:, :, ::-1])


def generate_output_image_name(config):
    prefix = os.path.basename(config["image"]).split(".")[0] + "_" + os.path.basename(config["style"]).split(".")[0]
    if "reconstruct_script" in config:
        suffix = f'_{config["optimizer"]}_{config["model"]}{config["image_format"][1]}'
    else:
        suffix = f'_{config["optimizer"]}_{config["model"]}_cw_{config["content_weight"]}_sw_{config["style_weight"]}_tv_{config["tv_weight"]}{config["image_format"][1]}'
    
    return prefix + suffix


def save_and_display(optimizing_image, dump_path, config, image_id, num_iterations, should_display=False):
    saving_freq = config["freq"]
    output_image = optimizing_image.squeeze(axis=0).to("cpu").detach().numpy()
    output_image = np.moveaxis(output_image, 0, 2)

    # Frequency -1 only saves last image after training
    # Frequency 1 saves all images while training
    if image_id == num_iterations-1 or (saving_freq > 0 and image_id % saving_freq == 0):
        image_format = config["image_format"]
        output_image_name = str(image_id).zfill(image_format[0]) + image_format[1] if saving_freq != -1 else generate_output_image_name(config)
        dump_image = np.copy(output_image)
        dump_image += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_image = np.clip(dump_image, 0, 255).astype("uint8")
        cv.imwrite(os.path.join(dump_path, output_image_name), dump_image[:, :, ::-1])

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(output_image)))
        plt.show()


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255

        return x

    else:
        raise ValueError(f"Expected numpy array got {type(x)}")


def prepare_model(model, device):
    if model == "vgg16":
        model = Vgg16(requires_grad=False, show_progress=True)

    elif model == "vgg19":
        model = Vgg19(requires_grad=False, show_progress=True)

    else:
        raise ValueError(f"{model} not supported")

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names

# Relation (gram) matrix
def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w

    return gram

def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


# Creating video from results
def create_video(results_path, image_format):
    output_file_name = "full_training_footage.mp4"
    fps = 30
    first_frame = 0
    num_frames = len(os.listdir(results_path))

    ffmpeg = "ffmpeg"
    if shutil.which(ffmpeg):
        image_name_format = "%" + str(image_format[0]) + "d" + image_format[1]
        pattern = os.path.join(results_path, image_name_format)
        output_video_path = os.path.join(results_path, output_file_name)

        trim_video_command = ["-start_number", str(first_frame), "-vframes", str(num_frames)]
        input_options = ["-r", str(fps), "-i", pattern]
        encoding_options = ["-c:v", "libx264", "-crf", "25", "-pix_fmt", "yuv420p"]
        subprocess.call([ffmpeg, *input_options, *trim_video_command, *encoding_options, output_video_path])
        
    else:
        print(f"{ffmpeg} not found in the system path, aborting")