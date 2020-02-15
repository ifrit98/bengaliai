import os
import cv2

import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


"""Read through for augmentation ideas:
  https://www.kaggle.com/corochann/bengali-seresnext-prediction-with-pytorch
  https://www.kaggle.com/iafoss/image-preprocessing-128x128
  https://www.kaggle.com/sgalella/bengali-ai-grapheme-classification-preprocessing
"""

TRAIN_DIR = 'data-raw/'

IMAGE_SIZE = 128
IMG_SIZE = IMAGE_SIZE
HEIGHT = 137
WIDTH = 236




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


def resize(df, size=IMG_SIZE, need_progress_bar=True):
    resized = {}
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH),(size, size))
            resized[df.index[i]] = image.reshape(-1)
    else:
        for i in range(df.shape[0]):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH),(size, size))
            resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
    

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
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


AUGMENTATIONS = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]



# taken from https://www.kaggle.com/iafoss/image-preprocessing-128x128
MEAN = [ 0.06922848809290576,  0.06922848809290576,  0.06922848809290576]
STD  = [ 0.20515700083327537,  0.20515700083327537,  0.20515700083327537]


# Questionable ? Test this to make sure normalization is correct
def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = np.expand_dims(image, -1)
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)


# TODO: Tuple out of range error in Image.fromarray(npa) 
## Must be a 2D numpy array (e.g. np.reshape(image, [137, 236]), not 3D)
def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  return np.asarray(pil_img) / 255.


def augment_and_mix(image, severity=1, width=3, depth=1, alpha=1.):
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
  ws = np.float32(np.random.dirichlet([alpha] * width))
  m  = np.float32(np.random.beta(alpha, alpha))
  
  mix = np.zeros(shape=[IMAGE_SIZE, IMAGE_SIZE, 1])
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(depth):
      op = np.random.choice(AUGMENTATIONS)
      image_aug = np.expand_dims(apply_op(image_aug, op, severity), -1)
    # Preprocessing commutes since all coefficients are convex
    #mix += ws[i] * normalize(image_aug)
    mix = np.add(mix, ws[i] * normalize(image_aug),casting="unsafe")
  mixed = (1 - m) * normalize(np.expand_dims(image, -1)) + m * mix
  return mixed


# img_filenames = os.listdir(TRAIN_DIR).sort()
# test = img_filenames[3]

def visualize(original_image, aug_image):
    fontsize = 18
    f, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original image', fontsize=fontsize)
    ax[1].imshow(aug_image,cmap='gray')
    ax[1].set_title('Augmented image', fontsize=fontsize)
    plt.show()
    

# TODO: Write functionality to quickly retreive shaped data (137x236)
# TODO: Choose best data import and augmentation methos to put in one linear script
# TODO: Create data pipeline to preprocess and potentially store training data
# TODO: Serializing examples with tf.Dataset vs on-the-fly augmentation?

from python.import_pd import *

# Visualize augmentations
# TODO: Stepthrough & debug augment_and_mix() so you can use this architecture

resized_df = resize(test1)

for idx, row in resized_df.iterrows():
    img = row.values
    img_og = np.reshape(img, [IMAGE_SIZE, IMAGE_SIZE]) # AUGMIX EXPECTS SQUARE IMAGE!
    img_aug = augment_and_mix(img_og)
    visualize(img_og, img_aug)
    
    
    
