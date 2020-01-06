# https://www.kaggle.com/gpreda/bengali-ai-handwritten-grapheme-getting-started

import os
import pandas as pd
import numpy as np
import PIL.Image
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from inst.python.data_tools import *

###############################################################################################
# https://www.kaggle.com/gpreda/bengali-ai-handwritten-grapheme-getting-started               #
###############################################################################################

for dirname, _, filenames in os.walk('data-raw/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

DATA_FOLDER = 'data-raw/'
train_df = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))
train_df.head()
train_df.shape

test_df = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
test_df.head()

class_map_df = pd.read_csv(os.path.join(DATA_FOLDER, 'class_map.csv'))
class_map_df.head()

sample_submission_df = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))
sample_submission_df.head()

start_time = time.time()
train_0_df = pd.read_parquet(os.path.join(DATA_FOLDER,'train_image_data_0.parquet'))
test_0_df  = pd.read_parquet(os.path.join(DATA_FOLDER,'test_image_data_0.parquet'))

# print(f"`train_image_data_0` read in {round(time.time()-start_time,2)} sec.")
print(f"`test_image_data_0` read in {round(time.time()-start_time,2)} sec.")

# train_0_df.shape
# train_0_df.head()

test_0_df.shape
test_0_df.head()


print(f"Train: unique grapheme roots: {train_df.grapheme_root.nunique()}")
print(f"Train: unique vowel diacritics: {train_df.vowel_diacritic.nunique()}")
print(f"Train: unique consonant diacritics: {train_df.consonant_diacritic.nunique()}")
print(f"Train: total unique elements: {train_df.grapheme_root.nunique() + train_df.vowel_diacritic.nunique() + train_df.consonant_diacritic.nunique()}")
print(f"Class map: unique elements: \n{class_map_df.component_type.value_counts()}")
print(f"Total combinations: {pd.DataFrame(train_df.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])).shape[0]}")

