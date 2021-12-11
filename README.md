## About
Creating fun images/videos by using Neural Style Transformation algorithms

## Installation
This project is built upon PyTorch using Python 3.8+
For GPU support install CUDA

All modules used are in file `requirements.txt`

Install: terminal -> `pip3 install -r requirements.txt`

## Usage
All parameters are setup in `config.json` file.
Parameters: 
    `image` - content image - name of content image, eg. `lion.jpg`
    `style` - style image - name of style image, eg. `cubist.jpg`
    `content_images_dir` - relative path to content images folder, eg. `data/content_images/`
    `style_images_dir` - relative path to style images folder, eg. `data/style_images/`
    `output_images_dir` - relative path to output images folder, eg. `data/output_images/`
    `freq` - how frequently will image be saved while training (`1` means after every step, `-1` means only final image will be saved)
    `video` - creating video from all training steps - `True` or `False`
    `height` - height of an image - int number, eg. `500`
    `image_format` - in what format will image be saved - leave first value to be 4, second value can be `.jpg`, `.png`, etc.
    `model` - which model will be used for optimizing - `vgg19` or `vgg16` 
    `optimizer` - which optimizer will model use - `adam` or `lbfgs` (lbfgs is better for this task, but in case your GPU doesn't have enough memory you can use Adam)
    `init_method` - initialization method for image - `content` or`random`
    `iterations` - number of iterations/epochs for training - int number, eg. `400`

