import numpy as np
from data_augmentation import *
import pathlib
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
root = '/media/khaledosama/CAF8CEC8F8CEB24D/Work/FCIS/Sem2/neural_network/project/fcis-cs-deeplearningcompetition'


'''
Loading dataset for each class divide them to train and validation set (80% train and 20% validation)
then augment each class images to make the images for each class hace the same number.
'''
def loading_dataset():
    label_to_index = {'airport_inside': 1,
                      'bakery': 2,
                      'bedroom': 3,
                      'greenhouse': 4,
                      'gym': 5,
                      'kitchen': 6,
                      'operating_room': 7,
                      'poolinside': 8,
                      'restaurant': 9,
                      'toystore': 10}

    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_CHANNELS = 3

    data_root = root
    data_root += '/train'
    data_root = pathlib.Path(data_root)
    classes = list(data_root.glob('*'))

    X_validation = []
    Y_validation = []
    ret_dataset = None
    k = True
    for folder in tqdm(classes):
        image_path = str(folder) + '/*.jpg'
        images = []
        labels = []
        for img in glob.glob(image_path):
            images.append(img)
            label = label_to_index[pathlib.Path(img).parent.name] - 1
            labels.append(label)

        X_train, X_valid, Y_train, Y_valid = train_test_split(images, labels, test_size=0.2)
        Y_train = labels_encoding(np.array(Y_train))
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        class_dataset = augmentation(dataset, pathlib.Path(images[0]).parent.name)
        if k:
            ret_dataset = class_dataset
            k = False
        else:
            ret_dataset = ret_dataset.concatenate(class_dataset)

        X_validation.extend(X_valid)
        Y_validation.extend(Y_valid)

    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation, dtype=np.int32)
    # print(Y_validation.shape)

    np.random.seed(5)
    p = np.random.permutation(X_validation.shape[0])
    X_validation = X_validation[p]
    Y_validation = Y_validation[p]

    Y_validation = labels_encoding(Y_validation)
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_validation, Y_validation))
    return ret_dataset, valid_dataset


'''
function to map each image path to an image
'''

def preprocess_image(image_path, label):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image, label
'''
after convert each image path to an image this function resize the images and convert the pixels value to range [-1:1]
'''
def resize_image(image, label):
  image = tf.image.resize_images(image, [224, 224])
  image = 2 * (image / 255.0) - 1.0
  return image, label
'''
convert label to onehot vector
'''
def labels_encoding(labels):
  ret = np.zeros((labels.shape[0], 10))
  ret[np.arange(labels.shape[0]), labels] = 1
  return ret

'''
loading testing dataset (which haven't labels)
'''
def load_testing_data():
  data_root = root
  data_root += '/test'
  data_root = pathlib.Path(data_root)
  all_img_paths = list(data_root.glob('*'))
  labels = []
  imgs = np.zeros((772, 224, 224, 3), dtype=np.float32)
  i = 0
  for file_name in all_img_paths:
    file_name = str(file_name)
    image_name = file_name.split('/')[-1]
    labels.append(image_name)
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224, 224))
    img = 2 * (img / 255.0) - 1.0
    imgs[i] = img
    i += 1
  return labels, tf.convert_to_tensor(imgs)