# Process:
# parquet -> pandas -> numpy -> batch generator -> tfrecord -> tfdataset

import pandas as pd
import numpy as np

import itertools

# Assumes cwd is project_dir (e.g. ~/internal/bengaliai)
from python.tfrecords.tfrecord_utils import record_generator
# from python.data_tools import normalize_simple


train_df_    = pd.read_csv('data/data-raw/train.csv')
test_df_     = pd.read_csv('data/data-raw/test.csv')
class_map_df = pd.read_csv('data/data-raw/class_map.csv')

NO_VOWELS     = len(train_df_['vowel_diacritic'].unique())
NO_CONSONANTS = len(train_df_['consonant_diacritic'].unique())
NO_GRAPHEMES  = len(train_df_['grapheme_root'].unique())

IMG_COLS = [str(i) for i in range(32332)]

# TODO: add transformations to generator so training dataset is deterministic
# Potential speed gain but will eat more Gb.
def apply_transformations():
  """See augmix.py and data_tools.py"""
  pass


def normalize_simple(image, maximum=255.0, v2=True):
  if v2 is True and len(image.shape) == 1:
    image = [image]
  
  normalized = np.asarray(map(lambda x: (x - np.mean(x)) / np.std(x), image)) \
    if v2 else (maximum - image).astype(np.float64) / maximum
  
  return normalized


# TOOD: add crop/resize/augmentation here
def img_generator(df, norm=False, scale=True, batch_size=8, onehot=False):
  
  for i in range(0, len(df), batch_size):
    image     = np.stack(df.iloc[i:i+batch_size][IMG_COLS].astype(np.int16).values, 0)
    grapheme  = np.stack(df.iloc[i:i+batch_size]['grapheme_root'].astype(np.int16), 0)
    vowel     = np.stack(df.iloc[i:i+batch_size]['vowel_diacritic'].astype(np.int16), 0)
    consonant = np.stack(df.iloc[i:i+batch_size]['consonant_diacritic'].astype(np.int16), 0)
    
    batch = {'image':     normalize_simple(image, scale=scale) if norm else image,
             'grapheme':  np.eye(NO_GRAPHEMES, dtype=np.int16)[grapheme] if onehot else grapheme,
             'vowel':     np.eye(NO_VOWELS, dtype=np.int16)[vowel] if onehot else vowel,
             'consonant': np.eye(NO_CONSONANTS, dtype=np.int16)[consonant] if onehot else consonant}
    yield batch


def create_generators(chain=True, rng=4):
  generators = []
  
  for i in range(rng):
    train_df = pd.merge(
    pd.read_parquet(f'data/data-raw/train_image_data_{i}.parquet'),
    train_df_, on='image_id'
    ).drop(['image_id'], axis=1)
    
    gen = img_generator(train_df)
    generators.append(gen)
    
  return generators if not chain else itertools.chain(generators)

  
## PROCESS:
# Call tfrecord apparatus on list of generators iteratively
# Apply image augmentation: before tfrecord generation or after, (lazily upon tfdataset creation)
# Split up into train/test data.  The "test_df_" is left blank intentionally. No labels.

def make_tfrecords(outdir='/tmp/tfrecords', chain=True, rng=1, num_batches=6276):
  ## Create tfdataset
  generators = create_generators(chain, rng=rng)
  
  if not chain:
    for gen in generators:
      record_generator(batch_generator=gen, output_dir=outdir, num_batches=num_batches)
  else:
    record_generator(batch_generator=generator, output_dir=outdir, num_batches=num_batches)
  
  print("Finished creating tfrecords in %s", outdir)
    
  return True


src_dir = '/home/jason/internal/bengali/data/data-tfrecord'
make_tfrecords(src_dir, False, 1)

