"""https://www.kaggle.com/pestipeti/bengali-quick-eda"""

# import cv2
# from tqdm import tqdm_notebook as tqdm
# import zipfile
# import io
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")
# import os
# import seaborn as sns
# import time
# import PIL.Image as Image
# import PIL.Image as Image
# import PIL.ImageDraw as ImageDraw
# import PIL.ImageFont as ImageFont
# import plotly.graph_objects as go


# Load all from data_tools.py and load_data.py
from inst.python.data_tools import *
# from inst.python.load_data import *

# train_ids, train_imgs = load_as_npa(TRAIN[0])
# train_full = load_all_data("train") # as np array (200840, 137, 236)
train_imgs = train_full
test_full  = load_all_data("test")  # as np array (12, 137, 236)
test_imgs = test_full

f, ax = plt.subplots(5, 5, figsize=(16, 8))
ax = ax.flatten()

for i in range(25):
    ax[i].imshow(train_imgs[i], cmap='Greys')
plt.show()


from inst.python.get_csv_data import *
train_df.head()
test_df.head()
class_map_df.head()
sample_submission_df.head()

# Grapheme roots
print("Number of unique grapheme_root: {}".format(train_df['grapheme_root'].nunique()))

fig = go.Figure(data=[go.Histogram(x=train_df['grapheme_root'])])
fig.update_layout(title_text='`grapheme_root` values')
fig.show()


# Most common grapheme root values
x = train_df['grapheme_root'].value_counts().sort_values()[-20:].index
y = train_df['grapheme_root'].value_counts().sort_values()[-20:].values
fig = go.Figure(data=[go.Bar(x=x, y=y)])
fig.update_layout(title_text='Most common `grapheme_root` values')
fig.show()



common_gr = class_map_df[(class_map_df['component_type'] == 'grapheme_root') & \
  (class_map_df['label'].isin(x))]['component']

f, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.flatten()

for i in range(20):
    ax[i].imshow(image_from_char(common_gr.values[i]), cmap='Greys')
plt.show()


# Least common root values

x = train_df['grapheme_root'].value_counts().sort_values()[:20].index
y = train_df['grapheme_root'].value_counts().sort_values()[:20].values
fig = go.Figure(data=[go.Bar(x=x, y=y)])
fig.update_layout(title_text='Least common `grapheme_root` values')
fig.show()



uncommon_gr = class_map_df[(class_map_df['component_type'] == 'grapheme_root') & \
  (class_map_df['label'].isin(x))]['component']

f, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.flatten()

plt.title('Uncommon `graheme_root` values')
for i in range(20):
    ax[i].imshow(image_from_char(uncommon_gr.values[i]), cmap='Greys')

plt.show()



# Vowel diacritic
train_df['vowel_diacritic'].nunique()

x = train_df['vowel_diacritic'].value_counts().sort_values().index
y = train_df['vowel_diacritic'].value_counts().sort_values().values
fig = go.Figure(data=[go.Bar(x=x, y=y)])
fig.update_layout(title_text='`vowel_diacritic` values')
fig.show()


vowels = class_map_df[(class_map_df['component_type'] == 'vowel_diacritic') & \
(class_map_df['label'].isin(x))]['component']

f, ax = plt.subplots(3, 5, figsize=(16, 8))
ax = ax.flatten()

for i in range(15):
    if i < len(vowels):
        ax[i].imshow(image_from_char(vowels.values[i]), cmap='Greys')
plt.show()


train_df['consonant_diacritic'].nunique()
x = train_df['consonant_diacritic'].value_counts().sort_values().index
y = train_df['consonant_diacritic'].value_counts().sort_values().values
fig = go.Figure(data=[go.Bar(x=x, y=y)])
fig.update_layout(title_text='`consonant_diacritic` values')
fig.show()

consonants = class_map_df[(class_map_df['component_type'] == 'consonant_diacritic') & \
  (class_map_df['label'].isin(x))]['component']

f, ax = plt.subplots(1, 7, figsize=(16, 8))
ax = ax.flatten()

for i in range(7):
    ax[i].imshow(image_from_char(consonants.values[i]), cmap='Greys')
plt.show()


## Similar Graphemes
plt.cla()
plt.clf()

# Get most common root
# Most common grapheme_root
gr_root_component = class_map_df[(class_map_df['component_type'] == 'grapheme_root') & \
  (class_map_df['label'] == 72)]['component']
plt.imshow(image_from_char(gr_root_component[72]), cmap='Greys')
plt.show()


# Digital variants of most common root (72)
train_df_short = train_df # train_df[train_df.index <= 50000]
samples = train_df_short[train_df['grapheme_root'] == 72].sample(n=25)
# samples.reset_index(drop=True, inplace=True)
f, ax = plt.subplots(5, 5, figsize=(16, 8))
ax = ax.flatten()
k = 0
for i, row in samples.iterrows():
    ax[k].imshow(image_from_char(row['grapheme']), cmap='Greys')
    k = k + 1
plt.show()


# Handwritten variants of the most common root (72)
f, ax = plt.subplots(2, 5, figsize=(16, 8))
ax = ax.flatten()
k = 0
samples_short = samples # samples[samples.index <= 50000]

for i, row in samples_short.iterrows():
    ax[k].imshow(train_imgs[i], cmap='Greys')
    k = k + 1
plt.show()



# Examples of grapheme root দ without vowel_diacritic and consonant_diacritic components
samples = train_df_short[
    (train_df_short['grapheme_root'] == 72) &
    (train_df_short['vowel_diacritic'] == 0) &
    (train_df_short['consonant_diacritic'] == 0)
].sample(n=25)

f, ax = plt.subplots(5, 5, figsize=(16, 8))
ax = ax.flatten()
k = 0
for i, row in samples.iterrows():
    ax[k].imshow(train_imgs[i], cmap='Greys')
    k = k + 1
plt.show()    
