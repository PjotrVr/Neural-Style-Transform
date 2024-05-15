## About

Creating fun images/videos by using Neural Style Transformation algorithms.

Some examples:

<p float="left">
  <img src="assets/dog.jpeg" width="100" />
  <span style="font-size: 30px; margin: 0 10px;">+</span>
  <img src="assets/wave.jpg" width="100" />
  <span style="font-size: 30px; margin: 0 10px;">=</span>
  <img src="assets/dog_wave.jpg" width="100" />
</p>

GIF of lion image being styled with cubist style: <br>
<img src="assets/lion_cubist_training_footage.gif"></img>

## Installation

This project is built upon PyTorch using Python 3.8+.

For GPU support install CUDA.

All modules used are in file `requirements.txt`.

Install: terminal -> `pip3 install -r requirements.txt`.

## Usage

All parameters are setup in `config.json` file.

Parameters: <br>

-   `image` - content image - name of content image, eg. `lion.jpg` <br>
-   `style` - style image - name of style image, eg. `cubist.jpg` <br>
-   `path_type` - type of path - `relative` or `absolute`
-   `content_images_dir` - path to content images folder, eg. `data/content_images/` <br>
-   `style_images_dir` - path to style images folder, eg. `data/style_images/` <br>
-   `output_images_dir` - path to output images folder, eg. `data/output_images/` <br>
-   `freq` - how frequently will image be saved while training (`1` means after every step, `2` after every 2 steps, `0` means only final image will be saved) <br>
-   `video` - creating video from all training steps - `True` or `False` (only possible if frequency is >= 1) <br>
-   `height` - height of an image - int number, eg. `500` <br>
-   `image_format` - in what format will image be saved - leave first value to be 4, second value can be `.jpg`, `.png`, etc. <br>
-   `model` - which model will be used for optimizing - `vgg16` or `vgg19` <br>
-   `optimizer` - which optimizer will model use - currently only `adam` is supported<br>
-   `init_method` - initialization method for image - `content` or`random` <br>
-   `iterations` - number of iterations/epochs for training - int number, eg. `400` <br>
