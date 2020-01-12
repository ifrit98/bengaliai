
import pyarrow.parquet as pq
# import pandas as pd
# from time import time

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
        

# def load_as_npa(file):
#     df = pd.read_parquet(file)
#     return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)



# Three ways to load data... find the most efficient
# start = time()
# test_ids0, test0 = load_as_npa(TEST[0])
# print("Loading as numpy array took:", (time() - start) %  60, "seconds")
# image_ids1, images1 = load_as_npa('data-raw/test_image_data_1.parquet')
# image_ids2, images2 = load_as_npa('data-raw/test_image_data_2.parquet')
# image_ids3, images3 = load_as_npa('data-raw/test_image_data_3.parquet')
# start = time()
# test1 = pd.read_parquet(TEST[1])
# print("Loading as pandas df took:", (time() - start) %  60, "seconds")
# df_train0 = pq.read_table("data-raw/train_image_data_0.parquet").to_pandas()
# df_train1 = pq.read_table("data-raw/train_image_data_1.parquet").to_pandas()
# df_train2 = pq.read_table("data-raw/train_image_data_2.parquet").to_pandas()
# df_train3 = pq.read_table("data-raw/train_image_data_3.parquet").to_pandas()

# df_test0 = pq.read_table("data-raw/test_image_data_1.parquet").to_pandas()
# df_test0 = load_as_npa("data-raw/test_image_data_0.parquet")
# df_test1 = pq.read_table("data-raw/test_image_data_1.parquet" ).to_pandas()
# start = time()
test2 = pq.read_table(TEST[0]).to_pandas()
# print("Loading as parquet and convert to pandas took:", (time() - start) %  60, "seconds")
# df_test3 = pq.read_table("data-raw/test_image_data_3.parquet").to_pandas()



  
