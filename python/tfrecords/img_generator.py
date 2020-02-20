#!usr/bin/python3

# Process:
# parquet -> pandas -> numpy -> batch generator -> tfrecord -> tfdataset

import pandas as pd
import numpy as np

import itertools

# Assumes cwd is project_dir (e.g. ~/internal/bengaliai)
from python.tfrecords.tfrecord_utils import record_generator
from python.data_tools import crop_resize, normalize, invert_and_reshape
from python.augmix2 import *


# TODO: do not batch here so can be done easily in tensorflow?
# TODO: Figure out how to keep modified images in generator
# TODO: experiment with no normalization but still crop

train_df_    = pd.read_csv('csv/train.csv')
test_df_     = pd.read_csv('csv/test.csv')
class_map_df = pd.read_csv('csv/class_map.csv')

NO_VOWELS     = len(train_df_['vowel_diacritic'].unique())
NO_CONSONANTS = len(train_df_['consonant_diacritic'].unique())
NO_GRAPHEMES  = len(train_df_['grapheme_root'].unique())

IMG_COLS = [str(i) for i in range(32332)]


def preprocess_generator(df, norm=True, img_size=128):
  crop = lambda x: crop_resize(x, size=img_size)
  
  for i in range(len(df)):
    image    = df.iloc[i][1:].astype(np.uint8).values
    inverted = invert_and_reshape(image)
    cropped  = crop(inverted)
    
    if norm:
      image = normalize(cropped)
    else:
      image = cropped
      
    yield image
  

# GOOD 92% acc from just this alone.  Don't change, but add augmentations to this
def img_generator(df, norm=True, batch_size=8, img_size=128, onehot=False):
  crop = lambda x: crop_resize(x, size=img_size)
  rng = range(batch_size)
  
  for i in range(0, len(df), batch_size):
    images    = np.stack(df.iloc[i:i+batch_size][IMG_COLS].astype(np.uint8).values, 0)
    grapheme  = np.stack(df.iloc[i:i+batch_size]['grapheme_root'].astype(np.uint8), 0)
    vowel     = np.stack(df.iloc[i:i+batch_size]['vowel_diacritic'].astype(np.uint8), 0)
    consonant = np.stack(df.iloc[i:i+batch_size]['consonant_diacritic'].astype(np.uint8), 0)
    
    inverted = np.asarray(list(map(invert_and_reshape, images)))
    cropped  = np.asarray(list(map(crop, inverted)))
    
    if norm:
      images = np.asarray(list(map(normalize, cropped))) 
    else:
      images = cropped
      
    batch = {'image':     images,
             'grapheme':  np.eye(NO_GRAPHEMES, dtype=np.uint8)[grapheme] if onehot else grapheme,
             'vowel':     np.eye(NO_VOWELS, dtype=np.uint8)[vowel] if onehot else vowel,
             'consonant': np.eye(NO_CONSONANTS, dtype=np.uint8)[consonant] if onehot else consonant}
    yield batch


def transform_generator(df, augment=True, norm=True, batch_size=8, severity=3, width=3, alpha=1., img_size=128):
  augmix  = lambda x: augment_and_mix(x, severity=severity, width=width, alpha=alpha)
  crop    = lambda x: crop_resize(x, size=img_size)
  scale   = lambda x: x / x.max()
  rng = range(batch_size)
  for i in range(0, len(df), batch_size):
    images    = np.stack(df.iloc[i:i+batch_size][IMG_COLS].astype(np.uint8).values)
    grapheme  = np.stack(df.iloc[i:i+batch_size]['grapheme_root'], 0)
    vowel     = np.stack(df.iloc[i:i+batch_size]['vowel_diacritic'], 0)
    consonant = np.stack(df.iloc[i:i+batch_size]['consonant_diacritic'], 0)
    inverted  = np.asarray(list(map(invert_and_reshape, images)))
    cropped   = np.asarray(list(map(crop, inverted)))
    augmented = np.asarray(list(map(augmix, cropped))) if augment else cropped
    normed    = np.asarray(list(map(normalize, augmented))) # TODO: small negative values OK?
    images    = np.asarray(list(map(scale, normed if norm else augmented)))
    batch = {'image':     images,
             'grapheme':  grapheme,
             'vowel':     vowel,
             'consonant': consonant}
    yield batch

# augment=True
# batch_size=8
# severity=3
# width=3
# alpha=1.
# size=128
# norm=True
# i=0
# rng=1

# TODO: Create loop for augmenting at different severity levels
# TODO: make a generator class inheriting from python generator?
# if augment=True, then create_generators() returns rng * 2 generators
def create_generators(rng=4, batch_size=8, augment=True, norm=True, severity=3, width=3, alpha=1., img_size=128):
  generators = []
  
  for i in range(rng):
    train_df = pd.merge(
    pd.read_parquet(f'data/data-raw/train_image_data_{i}.parquet'),
    train_df_, on='image_id'
    ).drop(['image_id'], axis=1)
    
    gen = img_generator(train_df, batch_size=batch_size, img_size=img_size)
    generators.append(gen)
    
    if augment:
      aug_gen = transform_generator(train_df, augment=augment, norm=norm, batch_size=batch_size, 
                                    severity=severity, width=width, alpha=alpha, img_size=img_size)
      generators.append(aug_gen)
      
  return gen if (rng == 1 and augment == False) else itertools.chain(generators)


def make_tfrecords(outdir='/tmp/tfrecords', batch_size=8, rng=4, num_batches=None, augment=True, norm=True):
  if num_batches is None: # img_per_pq / batch_size
    num_batches = int(np.floor(50208. / float(batch_size)))
  
  generators = create_generators(rng=rng, batch_size=batch_size, augment=augment, norm=norm)
  
  if rng > 1:
    for gen in generators:
      record_generator(batch_generator=gen, output_dir=outdir, num_batches=num_batches)
  else:
    record_generator(batch_generator=generators, output_dir=outdir, num_batches=num_batches)
  
  print("Finished creating tfrecords in %s", outdir)
  return True


#OUTDIR = SRC_DIR = '/home/jason/internal/bengali/data/data-tfrecord-aug'
#make_tfrecords(SRC_DIR)
# gen = create_generators(rng=2)



# df = pd.merge(pd.read_parquet('data/data-raw/train_image_data_0.parquet'),
#   train_df_, on='image_id').drop(['image_id'], axis=1)
# gen = img_generator(df)

# import matplotlib.pyplot as plt
# batch = gen.__next__()
# images = batch['image']
# batch_size=8
# fig, axs = plt.subplots(batch_size, 1, figsize=(10, batch_size))
# for idx in range(batch_size):
#   axs[idx].imshow(images[idx])
#   axs[idx].set_title('Original image')
#   axs[idx].axis('off')
# plt.show()


# n_imgs = 8
# fig, axs = plt.subplots(n_imgs, 4, figsize=(10, n_imgs))
# 
# for idx in range(n_imgs):
#   b = gen[0].__next__()
#   img = b['image'][0]
#   img_invert  = 255 - img.reshape(137, 236).astype(np.uint8) # correct color inversion
#   img_cropped = crop_resize(img_invert, size=128)
#   img_normed  = normalize(img_cropped, True)
#   img_normed2 = normalize(img_cropped)
#   axs[idx,0].imshow(img_invert)
#   axs[idx,0].set_title('Original image')
#   axs[idx,0].axis('off')
#   axs[idx,1].imshow(img_cropped)
#   axs[idx,1].set_title('Crop & resize')
#   axs[idx,1].axis('off')
#   axs[idx,2].imshow(img_normed)
#   axs[idx,2].set_title('normalized') # MEAN, STD are of whole distribution
#   axs[idx,2].axis('off')
#   axs[idx,3].imshow(img_normed2)
#   axs[idx,3].set_title('normalized_v2') # MEAN, STD are of each image (start here!)
#   axs[idx,3].axis('off')
#   
# plt.show()
# 
