# Process:
# parquet -> pandas -> numpy -> batch generator -> tfrecord -> tfdataset

import pandas as pd
import numpy as np


train_df_ = pd.read_csv('data/data-raw/train.csv')
test_df_ = pd.read_csv('data/data-raw/test.csv')
class_map_df = pd.read_csv('data/data-raw/class_map.csv')


IMG_SIZE = 128
N_CHANNELS = 1
HEIGHT = 137
WIDTH = 236
BATCH = 3


test_df = pd.merge(
  pd.read_parquet(f'data/data-raw/test_image_data_{0}.parquet'),
  test_df_, on='image_id'
).drop(['image_id'], axis=1)


df0 = pd.read_parquet(f'data/data-raw/test_image_data_{0}.parquet')
# df1 = pd.read_parquet(f'data/data-raw/test_image_data_{1}.parquet')
# df2 = pd.read_parquet(f'data/data-raw/test_image_data_{2}.parquet')
# df3 = pd.read_parquet(f'data/data-raw/test_image_data_{3}.parquet')


# df = df0.merge(df1)
df = df0


def generator(df):
  # grab only pixel values
  # x = df.iloc[0][:-2].astype(np.int16)

  # iterate over pandas df 
  for idx, row in df.iterrows():
    if idx % BATCH == 0:
      pass
  # create batch
  batch = {'image': np.stack([df.iloc[i][1:].astype(np.int16).values 
                        for i in range(BATCH)], 0),
           'grapheme': ,
           'vowel':,
           'consonant':}

  
# Split up into train/test data.  The "test_df_" is left blank intentionally. No labels.
