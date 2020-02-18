import pandas as pd

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


test0 = pd.read_parquet(TEST[0])
test1 = pd.read_parquet(TEST[1])
test2 = pd.read_parquet(TEST[2])
test3 = pd.read_parquet(TEST[3])

test_data = [test0, test1, test2, test3]
    

def merge_dfs(dfs):
  """Accepts a list of dataframes to merge together"""
  full_df = pd.DataFrame()
  for df in dfs:
      full_df = df.merge(full_df, how='outer', left_index=True, right_index=True)
  return full_df



def load_test_data(files):
  dfs = []
  for i in range(len(files)):
    dfs.append(pd.read_parquet(files[i]))
  df = merge_dfs(dfs)
  return df


# test_df = load_test_data(TEST[:2])
