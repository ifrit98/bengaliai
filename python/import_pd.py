
import pandas as pd

HEIGHT = 137
WIDTH  = 236
BATCH  = 128


TRAIN = ['data-raw/train_image_data_0.parquet',
         'data-raw/train_image_data_1.parquet',
         'data-raw/train_image_data_2.parquet',
         'data-raw/train_image_data_3.parquet']
         
TEST = ['data-raw/test_image_data_0.parquet',
        'data-raw/test_image_data_1.parquet',
        'data-raw/test_image_data_2.parquet',
        'data-raw/test_image_data_3.parquet']
        


test1 = pd.read_parquet(TEST[1])
test1 = test1.drop(columns=['image_id'])
df = test1

  
