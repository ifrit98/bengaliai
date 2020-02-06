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
  pd.read_parquet(f'data/data-raw/test_image_data_{i}.parquet'),
  test_df_, on='image_id'
).drop(['image_id'], axis=1)


df = pd.read_parquet(f'data/data-raw/test_image_data_{i}.parquet')

def generator(df):
  # grab only pixel values
  # x = df.iloc[0][:-2].astype(np.int16)

  # iterate over pandas df 
  for idx, row in df.iterrows():
    if idx % 3 == 0:
      pass
  # create batch
  batch = {"image" : np.stack([df.iloc[i][1:].astype(np.int16).values 
                        for i in range(BATCH)], 0)}

  
