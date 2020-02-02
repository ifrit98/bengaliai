
import tensorflow as tf
from python.DataGenerator import MultiOutputDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2


from sklearn.model_selection import train_test_split
from tqdm import tqdm


train_df_ = pd.read_csv('data-raw/train.csv')
test_df_ = pd.read_csv('data-raw/test.csv')
class_map_df = pd.read_csv('data-raw/class_map.csv')
sample_sub_df = pd.read_csv('data-raw/sample_submission.csv')



IMG_SIZE = 64
N_CHANNELS = 1
HEIGHT = 137
WIDTH = 236

batch_size = 32
epochs = 1



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

    
def one_hot(df, colname='grapheme_root', dtype=tf.int8):
  x = tf.one_hot(df[colname], depth=len(df[colname].unique()), dtype = dtype)
  return x


# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_3_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_4_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_5_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)


from python.model import load_model
model = load_model()

from time import time


i = 1
histories = []
for i in range(4):
  
    start = time()
  
    train_df = pd.merge(
      pd.read_parquet(f'data-raw/train_image_data_{i}.parquet'),
      train_df_, on='image_id'
    ).drop(['image_id'], axis=1)
    
    
    diff = time() - start
    print("Loading training data took", diff, "seconds")
    # # Visualize few samples of current training dataset
    # fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
    # count=0
    # for row in ax:
    #     for col in row:
    #         col.imshow(resize(
    #           train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis=1).iloc[[count]],
    #           need_progress_bar=False).values.reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))
    #         count += 1
    # plt.show()

    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis=1)
    X_train = resize(X_train)/255
    
    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    # # As Tensors
    # Y_train_root_onehot = one_hot(train_df, 'grapheme_root')
    # Y_train_vowel_onehot = one_hot(train_df, 'vowel_diacritic')
    # Y_train_consonant_onehot = one_hot(train_df, 'consonant_diacritic')
  
    
    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values
    
    
    np.savez_compressed("train_root_labels{i}", train_root_labels=Y_train_root)



    print(f'Training images: {X_train.shape}')
    print(f'Training labels root: {Y_train_root.shape}')
    print(f'Training labels vowel: {Y_train_vowel.shape}')
    print(f'Training labels consonants: {Y_train_consonant.shape}')

    # Divide the data into training and validation set
    x_train, x_test, y_train_root, y_test_root, y_train_vowel, \
    y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(
      X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
    del train_df
    del X_train
    del Y_train_root, Y_train_vowel, Y_train_consonant

    # Data augmentation for creating more training data
    datagen = MultiOutputDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    # This will just calculate parameters required to augment the given data. This won't perform any augmentations
    datagen.fit(x_train)

    # Fit the model
    history = model.fit_generator(
      datagen.flow(x_train, {'dense_2': y_train_root, 'dense_3': y_train_vowel, 'dense_4': y_train_consonant}, 
      batch_size=batch_size),
      epochs = epochs, 
      validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
      steps_per_epoch=x_train.shape[0] // batch_size, 
      callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant]
    )

    histories.append(history)
    
    # Delete to reduce memory usage
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant
    gc.collect()
