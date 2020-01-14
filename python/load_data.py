import pandas as pd

TRAIN = ['data-raw/train_image_data_0.parquet',
         'data-raw/train_image_data_1.parquet',
         'data-raw/train_image_data_2.parquet',
         'data-raw/train_image_data_3.parquet']
         
TEST = ['data-raw/test_image_data_0.parquet',
        'data-raw/test_image_data_1.parquet',
        'data-raw/test_image_data_2.parquet',
        'data-raw/test_image_data_3.parquet']



train_df_ = pd.read_csv('data-raw/train.csv')
test_df_ = pd.read_csv('data-raw/test.csv')
class_map_df = pd.read_csv('data-raw/class_map.csv')
sample_sub_df = pd.read_csv('data-raw/sample_submission.csv')


# test0 = pd.read_parquet(TEST[0])
# test1 = pd.read_parquet(TEST[1])
# test2 = pd.read_parquet(TEST[2])
# test3 = pd.read_parquet(TEST[3])


def load_all_and_merge_df(files, data=[TRAIN[0]]):
  df = pd.DataFrame()
  for i in range(len(files)):
    _df = pd.merge(
      pd.read_parquet(data[i]),
      train_df_, on='image_id'
    ).drop(['image_id'], axis=1)
    
    df = df.append(_df)
    
  return df


# test_df = load_all_and_merge_df(TEST)
train_df = load_all_and_merge_df([TRAIN[0]])
