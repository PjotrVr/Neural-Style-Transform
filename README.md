## About
Creating fun images/videos by using Neural Style Transformation algorithms

## Installation
This project is built upon PyTorch using Python 3.8+

For GPU support install CUDA

All modules used are in file `requirements.txt`

Install: terminal -> `pip3 install -r requirements.txt`

## Usage
All parameters are setup in `config.json` file

Parameters: <br>
* `image` - content image - name of content image, eg. `lion.jpg` <br>
* `style` - style image - name of style image, eg. `cubist.jpg` <br>
* `path_type` - type of path - `relative` or `absolute`
* `content_images_dir` - path to content images folder, eg. `data/content_images/` <br>
* `style_images_dir` - path to style images folder, eg. `data/style_images/` <br>
* `output_images_dir` - path to output images folder, eg. `data/output_images/` <br>
* `freq` - how frequently will image be saved while training (`1` means after every step, `-1` means only final image will be saved) <br>
* `video` - creating video from all training steps - `True` or `False` (in case you want video freq mustn't be -1) <br>
* `height` - height of an image - int number, eg. `500` <br>
* `image_format` - in what format will image be saved - leave first value to be 4, second value can be `.jpg`, `.png`, etc. <br>
* `model` - which model will be used for optimizing - `vgg16` or `vgg19` <br>
* `optimizer` - which optimizer will model use - `adam` or `lbfgs` (lbfgs is better for this task, but in case your GPU doesn't have enough memory you can use Adam) <br>
* `init_method` - initialization method for image - `content` or`random` <br>
* `iterations` - number of iterations/epochs for training - int number, eg. `400` <br>

