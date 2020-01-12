
from tensorflow import one_hot
import pandas as pd
from numpy import savez_compressed

from tqdm import tqdm


train_df_ = pd.read_csv('data-raw/train.csv')
test_df_ = pd.read_csv('data-raw/test.csv')
class_map_df = pd.read_csv('data-raw/class_map.csv')
sample_sub_df = pd.read_csv('data-raw/sample_submission.csv')



IMG_SIZE = 64
N_CHANNELS = 1
HEIGHT = 137
WIDTH = 236



def resize(df, size=IMG_SIZE, need_progress_bar=True):
    resized = {}
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(HEIGHT,WIDTH),(size, size))
            resized[df.index[i]] = image.reshape(-1)
    else:
        for i in range(df.shape[0]):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(HEIGHT,WIDTH),(size, size))
            resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
    
    
def get_dummies(df):
    cols = []
    for col in df:
        cols.append(pd.get_dummies(df[col].astype(str)))
    return pd.concat(cols, axis=1)

# Convert to one_hot for use with tfdataset
def one_hot(df, colname='grapheme_root', dtype=tf.int8):
  x = tf.one_hot(df[colname], depth=len(df[colname].unique()), dtype = dtype)
  return x



for i in range(1,4):
  train_df = pd.merge(
    pd.read_parquet(f'data-raw/train_image_data_{i}.parquet'),
    train_df_, on='image_id'
  ).drop(['image_id'], axis=1)
  X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis=1)
  X_train = resize(X_train) / 255
  X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
  Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
  Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
  Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values
  savez_compressed(f"data-npz/train_images{i}", train_images=X_train)
  savez_compressed(f"data-npz/train_root_labels{i}", train_root_labels=Y_train_root)
  savez_compressed(f"data-npz/train_vowel_labels{i}", train_vowel_labels=Y_train_vowel)
  savez_compressed(f"data-npz/train_consonant_labels{i}", train_consonant_labels=Y_train_consonant)
