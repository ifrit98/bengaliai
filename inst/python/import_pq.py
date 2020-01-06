
# import pyarrow.parquet as pq
import pandas as pd


HEIGHT = 137
WIDTH  = 236

def load_as_npa(file):
    df = pd.read_parquet(file)
    return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

image_ids0, images0 = load_as_npa('data-raw/test_image_data_0.parquet')

# df_train0 = pq.read_table("data-raw/train_image_data_0.parquet").to_pandas()
# df_train1 = pq.read_table("data-raw/train_image_data_1.parquet").to_pandas()
# df_train2 = pq.read_table("data-raw/train_image_data_2.parquet").to_pandas()
# df_train3 = pq.read_table("data-raw/train_image_data_3.parquet").to_pandas()

# df_test0 = pq.read_table("data-raw/test_image_data_1.parquet").to_pandas()
# df_test0 = load_as_npa("data-raw/test_image_data_0.parquet")
# df_test1 = pq.read_table("data-raw/test_image_data_1.parquet" ).to_pandas()
# df_test2 = pq.read_table("data-raw/test_image_data_2.parquet").to_pandas()
# df_test3 = pq.read_table("data-raw/test_image_data_3.parquet").to_pandas()


