
import pandas as pd
from python.tfrecords.img_generator import preprocess_generator    


TRAIN = ['data/data-raw/train_image_data_0.parquet',
         'data/data-raw/train_image_data_1.parquet',
         'data/data-raw/train_image_data_2.parquet',
         'data/data-raw/train_image_data_3.parquet']
         
TEST = ['data/data-raw/test_image_data_0.parquet',
        'data/data-raw/test_image_data_1.parquet',
        'data/data-raw/test_image_data_2.parquet',
        'data/data-raw/test_image_data_3.parquet']



# train_df_ = pd.read_csv('csv/train.csv')
test_df_ = pd.read_csv('csv/test.csv')
class_map_df = pd.read_csv('csv/class_map.csv')
sample_sub_df = pd.read_csv('csv/sample_submission.csv')


# testgen = preprocess_generator(pd.read_parquet(TEST[0]))

# Why long vector not supported error?
def test_generator(filepaths=TEST):
  for fp in filepaths:
    df = pd.read_parquet(fp)
    gen = preprocess_generator(df)
    
    imgs = []
    for img in gen:
      imgs.append(img)
      
    yield np.asarray(imgs)


# Geberates all and returns a nparray with all 12 test images
def load_test_data(filepaths=TEST):
  imgs = []
  
  for fp in filepaths:
    df = pd.read_parquet(fp)
    gen = preprocess_generator(df)
    
    for img in gen:
      imgs.append(img)
      
  return np.asarray(imgs)


def merge_dfs(dfs):
  """Accepts a list of dataframes to merge together"""
  full_df = pd.DataFrame()
  for df in dfs:
      full_df = df.merge(full_df, how='outer', left_index=True, right_index=True)
  return full_df

