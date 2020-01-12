
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
        

def load_as_npa(file):
    df = pd.read_parquet(file)
    return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)


test_ids0, test0 = load_as_npa(TEST[0])
npa = test0


  
