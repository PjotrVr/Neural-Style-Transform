from collections import namedtuple
import torch
from torchvision import models

# These architectures are taken from https://github.com/gordicaleksa/pytorch-neural-style-transfer
# Check Aleksa out, he has a lot of good content

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        pretrained_features = models.vgg16(pretrained=True, progress=show_progress).features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_feature_maps_index = 1  # relu2_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # all layers used for style representation

        self.slice_1 = torch.nn.Sequential()
        self.slice_2 = torch.nn.Sequential()
        self.slice_3 = torch.nn.Sequential()
        self.slice_4 = torch.nn.Sequential()

        for x in range(4):
            self.slice_1.add_module(str(x), pretrained_features[x])

        for x in range(4, 9):
            self.slice_2.add_module(str(x), pretrained_features[x])

        for x in range(9, 16):
            self.slice_3.add_module(str(x), pretrained_features[x])

        for x in range(16, 23):
            self.slice_4.add_module(str(x), pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()

        pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features
        if use_relu:  # use relu or as in original paper conv layers
            self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
            self.offset = 1

        else:
            self.layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
            self.offset = 0

        self.content_feature_maps_index = 4  # conv4_2

        # all layers used for style representation except conv4_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(self.content_feature_maps_index)  # conv4_2

        self.slice_1 = torch.nn.Sequential()
        self.slice_2 = torch.nn.Sequential()
        self.slice_3 = torch.nn.Sequential()
        self.slice_4 = torch.nn.Sequential()
        self.slice_5 = torch.nn.Sequential()
        self.slice_6 = torch.nn.Sequential()

        for x in range(1+self.offset):
            self.slice_1.add_module(str(x), pretrained_features[x])

        for x in range(1+self.offset, 6+self.offset):
            self.slice_2.add_module(str(x), pretrained_features[x])

        for x in range(6+self.offset, 11+self.offset):
            self.slice_3.add_module(str(x), pretrained_features[x])

        for x in range(11+self.offset, 20+self.offset):
            self.slice_4.add_module(str(x), pretrained_features[x])

        for x in range(20+self.offset, 22):
            self.slice_5.add_module(str(x), pretrained_features[x])

        for x in range(22, 29+self.offset):
            self.slice_6.add_module(str(x), pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice_1(x)
        layer1_1 = x

        x = self.slice_2(x)
        layer2_1 = x

        x = self.slice_3(x)
        layer3_1 = x

        x = self.slice_4(x)
        layer4_1 = x

        x = self.slice_5(x)
        conv4_2 = x

        x = self.slice_6(x)
        layer5_1 = x

        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out
