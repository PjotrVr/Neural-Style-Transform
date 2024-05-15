import numpy as np
import os
import json

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable

import utils


def build_loss(neural_net, optimizing_image, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_image)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction="mean")(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]

    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])

    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_image)

    total_loss = config["content_weight"] * content_loss + config["style_weight"] * style_loss + config["tv_weight"] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    def tuning_step(optimizing_image):
        # Calculate loss
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_image, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step


def neural_style_transfer(config):
    content_image_path = os.path.join(config["content_images_dir"], config["image"])
    style_image_path = os.path.join(config["style_images_dir"], config["style"])

    output_dir_name = f"styled_{os.path.split(content_image_path)[1].split('.')[0]}_{os.path.split(style_image_path)[1].split('.')[0]}"

    dump_path = os.path.join(config["output_images_dir"], output_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = utils.prepare_image(content_image_path, config["height"], device)
    style_image = utils.prepare_image(style_image_path, config["height"], device)

    if config["init_method"] == "random":
        #white_noise_image = np.random.uniform(-90., 90., content_image.shape).astype(np.float32)
        gaussian_noise_image = np.random.normal(loc=0, scale=90., size=content_image.shape).astype(np.float32)
        init_image = torch.from_numpy(gaussian_noise_image).float().to(device)
    
    elif config["init_method"] == "content":
        init_image = content_image
    
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_image_resized = utils.prepare_image(style_image_path, np.asarray(content_image.shape[2:]), device)
        init_image = style_image_resized

    optimizing_image = Variable(init_image, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config["model"], device)
    print(f'Using {config["model"]} model and {config["optimizer"]} optimizer')

    content_image_set_of_feature_maps = neural_net(content_image)
    style_image_set_of_feature_maps = neural_net(style_image)

    target_content_representation = content_image_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_image_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    num_iterations = int(config["iterations"])

    if config["optimizer"] == "adam":
        optimizer = Adam((optimizing_image,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_iterations):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_image)
            
            with torch.no_grad():
                print(f'Epoch: {cnt+1}/{num_iterations}, Total loss={total_loss.item():12.4f}, Content loss={config["content_weight"] * content_loss.item():12.4f}, Style loss={config["style_weight"] * style_loss.item():12.4f}, Tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_display(optimizing_image, dump_path, config, cnt, num_iterations, should_display=False)
    
    elif config["optimizer"] == "lbfgs":
        optimizer = LBFGS((optimizing_image,), max_iter=num_iterations, line_search_fn="strong_wolfe")
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_image, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            
            if total_loss.requires_grad:
                total_loss.backward()

            with torch.no_grad():
                print(f'Epoch: {cnt+1}/{num_iterations}, Total loss={total_loss.item():12.4f}, Content loss={config["content_weight"] * content_loss.item():12.4f}, Style loss={config["style_weight"] * style_loss.item():12.4f}, Tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_display(optimizing_image, dump_path, config, cnt, num_iterations, should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path


if __name__ == "__main__":
    # Loading config
    with open("config.json", "r", encoding="utf8") as json_file:
        optimization_config = json.load(json_file)
    
    if optimization_config["path_type"] == "relative":
        # Fixing paths
        optimization_config["content_images_dir"] = os.path.join(os.path.dirname(__file__), optimization_config["content_images_dir"])
        optimization_config["style_images_dir"] = os.path.join(os.path.dirname(__file__), optimization_config["style_images_dir"])
        optimization_config["output_images_dir"] = os.path.join(os.path.dirname(__file__), optimization_config["output_images_dir"])

    results_path = neural_style_transfer(optimization_config)

    if optimization_config["video"]:
        utils.create_video(results_path, optimization_config["image_format"])
    