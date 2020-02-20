# https://www.kaggle.com/haqishen/augmix-based-on-albumentations

import os
import cv2
import numpy as np
import pandas as pd
#import albumentations
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

import itertools

# Assumes cwd is project_dir (e.g. ~/internal/bengaliai)
# from python.tfrecords.tfrecord_utils import record_generator
from python.data_tools import crop_resize, normalize, invert_and_reshape


HEIGHT = 137
WIDTH = 236
SIZE = 128
IMG_COLS = [str(i) for i in range(32332)]

train_df_    = pd.read_csv('csv/train.csv')
test_df_     = pd.read_csv('csv/test.csv')
class_map_df = pd.read_csv('csv/class_map.csv')

NO_VOWELS     = len(train_df_['vowel_diacritic'].unique())
NO_CONSONANTS = len(train_df_['consonant_diacritic'].unique())
NO_GRAPHEMES  = len(train_df_['grapheme_root'].unique())


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
 
# def normalize(image):
#     """Normalize input image channel-wise to zero mean and unit variance."""
#     return image - 127

# Expects uint8 values in range (0, 255)
def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255) # clip values
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws  = np.float32(np.random.dirichlet([alpha] * width))
    m   = np.float32(np.random.beta(alpha, alpha))
    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        mix += ws[i] * image_aug
    mixed = (1 - m) * image + m * mix
    return mixed


# augment = True
# batch_size=8
# severity=3
# width=3
# alpha=1.
# size=128
# norm=True
# i=0
# def transform_generator(df, augment=False, norm=True, batch_size=8, severity=3, width=3, alpha=1., size=128):
#   augmix  = lambda x: augment_and_mix(x, severity=severity, width=width, alpha=alpha)
#   crop    = lambda x: crop_resize(x, size=size)
#   scale   = lambda x: x / x.max()
#   rng = range(batch_size)
#   
#   for i in range(0, len(df), batch_size):
#     images    = np.stack(df.iloc[i:i+batch_size][IMG_COLS].astype(np.uint8).values)
#     grapheme  = np.stack(df.iloc[i:i+batch_size]['grapheme_root'], 0)
#     vowel     = np.stack(df.iloc[i:i+batch_size]['vowel_diacritic'], 0)
#     consonant = np.stack(df.iloc[i:i+batch_size]['consonant_diacritic'], 0)
#     inverted  = np.asarray(list(map(invert_and_reshape, images)))
#     cropped   = np.asarray(list(map(crop, inverted)))
#     augmented = np.asarray(list(map(augmix, cropped))) if augment else cropped
#     normed    = np.asarray(list(map(normalize, augmented))) # TODO: small negative values OK?
#     images    = np.asarray(list(map(scale, normed if norm else augmented)))
#     
#     batch = {'image':     images,
#              'grapheme':  grapheme,
#              'vowel':     vowel,
#              'consonant': consonant}
#     yield batch


def transform_and_plot(dataset, severity=3, width=3, depth=-1, alpha=1., size=128):
  from pylab import rcParams
  rcParams['figure.figsize'] = 20, 10
  f, axarr = plt.subplots(3,2)
  for i in range(3):
    for p in range(2):
      idx = np.random.randint(0, len(dataset))
      img = dataset.iloc[idx][:-4].values.astype(np.uint8)
      label = dataset.iloc[idx][-4:-1].values.astype(np.uint8)
      img = img.reshape(137, 236) #.astype(np.float32)
      img = cv2.resize(img, (size, size))
      img = augment_and_mix(img, severity=3, width=3, depth=-1, alpha=1.)
      # img = np.expand_dims(img, -1)
      # img = np.repeat(img, 3, -1)  # 1ch to 3ch (H, W, C)
      img /= 255
      img = 1 - img
      axarr[i][p].imshow(img)
      axarr[i][p].set_title(idx)
  plt.show()


# dataset = pd.merge(
#   pd.read_parquet('data/data-raw/train_image_data_0.parquet'),
#   train_df_, on='image_id'
#   ).drop(['image_id'], axis=1)
#   
# gen = transform_generator(dataset, True)
#   
# image = dataset.iloc[0][:-4].values.astype(np.uint8)
# label = dataset.iloc[0][-4:-1].values.astype(np.int32)

# Look at transformation effects
# transform_and_plot(dataset, severity=3, width=3, alpha=1.)
# 
# # DIRTY
# transform_and_plot(dataset, severity=7, width=7, alpha=5.)
# 
# # NASTY
# transform_and_plot(dataset, severity=12, width=12, alpha=9.)


##
# TODO: create aug images generators, then chain in TFRECORD serialization!
# set generators at increasing severity params (1, 3, 5, 7, etc.)
##


# data_dir = 'data/data-raw/'
# files_train = [f'train_image_data_{fid}.parquet' for fid in range(1)]

# def read_data(files):
#     tmp = []
#     for f in files:
#         F = os.path.join(data_dir, f)
#         data = pd.read_parquet(F)
#         tmp.append(data)
#     tmp = pd.concat(tmp)
#     data = tmp.iloc[:, 1:].values
#     return data
# 
# train data
# data_train = read_data(files_train)
