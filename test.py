import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as vgg
import tensorflow.contrib.slim as slim


vgg.resnet_v2_152()
predictions = vgg.vgg_16()


