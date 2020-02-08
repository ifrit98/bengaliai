# Process:
# parquet -> pandas -> numpy -> batch generator -> tfrecord -> tfdataset

import pandas as pd
import numpy as np



train_df_ = pd.read_csv('data/data-raw/train.csv')
test_df_ = pd.read_csv('data/data-raw/test.csv')
class_map_df = pd.read_csv('data/data-raw/class_map.csv')


HEIGHT = 137
WIDTH = 236

IMG_COLS = [str(i) for i in range(32332)]


# TODO: add transformations to generator so training dataset is deterministic
# Potential speed gain but will eat more Gb.
def apply_transformations():
  """See augmix.py and data_tools.py"""
  pass


def img_generator(df, batch_size=8):
  for i in range(0, len(df), batch_size):
    batch = {'image':     np.stack(df.iloc[i:i+batch_size][IMG_COLS].astype(np.int16).values, 0),
             'grapheme':  np.stack(df.iloc[i:i+batch_size]['grapheme_root'].astype(np.int16), 0),
             'vowel':     np.stack(df.iloc[i:i+batch_size]['vowel_diacritic'].astype(np.int16), 0),
             'consonant': np.stack(df.iloc[i:i+batch_size]['consonant_diacritic'].astype(np.int16), 0)}
    yield batch


def create_generators():
  generators = []
  
  for i in range(4):
    train_df = pd.merge(
    pd.read_parquet(f'data/data-raw/train_image_data_{i}.parquet'),
    train_df_, on='image_id'
    ).drop(['image_id'], axis=1)
    
    gen = img_generator(train_df)
    generators.append(gen)
    
  return generators

  
## PROCESS:
# Call tfrecord apparatus on list of generators iteratively
# Apply image augmentation: before tfrecord generation or after, (lazily upon tfdataset creation)
# Split up into train/test data.  The "test_df_" is left blank intentionally. No labels.

## Create tfdataset
from tfrecord_utils import record_generator

generators = create_generators()

for gen in generators:
  record_generator(batch_generator=gen, output_dir='/tmp/tfrecords', num_batches=6276)
