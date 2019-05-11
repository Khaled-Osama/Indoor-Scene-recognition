import tensorflow as tf
from Data_Utils import *
def gausian_noise(img, label):
  img /= 255.0
  noise = tf.random_normal(tf.shape(img), stddev=0.07)
  image = img + noise
  return image, label

def increase_brightness(img, label):
  img /= 255.0
  img = tf.image.adjust_brightness(img, 0.2)
  return img, label

def decrease_brightness(img, label):
  img /= 255.0
  img = tf.image.adjust_brightness(img, -0.2)
  return img, label

def random_crop(img, label):
  img /= 255.0
  shape = tf.shape(img)
  img = tf.image.random_crop(img, (tf.math.minimum(shape[0], 224), tf.math.minimum(shape[1], 224), 3))
  return img, label

def random_saturation(img, label):
  img /= 255.0
  img = tf.image.random_saturation(img, 0, 0.5)
  return img, label

def flipping(image, label):
  image /= 255.0
  image = tf.image.flip_left_right(image)
  return image, label

'''
function to map each image path to an image
'''

def preprocess_image(image_path, label):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image, label


def augmentation(orig_dataset, class_name):
    augmentations = [gausian_noise, increase_brightness, random_crop, random_saturation, decrease_brightness]
    num_operations = {'airport_inside': 0,
                      'bakery': 1,
                      'bedroom': 0,
                      'greenhouse': 5,
                      'gym': 2,
                      'kitchen': 0,
                      'operating_room': 4,
                      'poolinside': 3,
                      'restaurant': 1,
                      'toystore': 1}

    it = 1

    orig_dataset = orig_dataset.map(preprocess_image)

    comb_dataset = orig_dataset

    while it <= num_operations[class_name]:
        aug_dataset = orig_dataset
        aug_dataset = aug_dataset.map(augmentations[it - 1])
        comb_dataset = comb_dataset.concatenate(aug_dataset)
        it += 1

    return comb_dataset