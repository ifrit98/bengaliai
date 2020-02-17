import tensorflow as tf
import cv2
from tqdm import tqdm_notebook as tqdm
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import time
import PIL.Image as Image
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import plotly.graph_objects as go



HEIGHT = 137
WIDTH  = 236
SIZE   = 128
IMG_SIZE = SIZE

TRAIN = ['data/data-raw/train_image_data_0.parquet',
         'data/data-raw/train_image_data_1.parquet',
         'data/data-raw/train_image_data_2.parquet',
         'data/data-raw/train_image_data_3.parquet']
         
TEST = ['data/data-raw/test_image_data_0.parquet',
        'data/data-raw/test_image_data_1.parquet',
        'data/data-raw/test_image_data_2.parquet',
        'data/data-raw/test_image_data_3.parquet']

OUT_TRAIN = 'train.zip'
OUT_TEST  = 'test.zip'



EAGER = "<class 'tensorflow.python.framework.ops.EagerTensor'>"


def load_as_npa(file=TEST[0]):
    df = pd.read_parquet(file)
    return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)


def load_all_data(datatype="test"):
    DATA = TRAIN if datatype.lower() == "train" else TEST
    dfs = []
    for i in range(len(DATA)):
        idx, imgs = load_as_npa(DATA[i])
        dfs.append(imgs)
    return np.concatenate(dfs, axis=0)
    

def resize(df, size=IMG_SIZE, need_progress_bar=True):
    resized = {}
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH), (size, size))
            resized[df.index[i]] = image.reshape(-1)
    else:
        for i in range(df.shape[0]):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH), (size, size))
            resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
    
    
# taken from https://www.kaggle.com/iafoss/image-preprocessing-128x128
MEAN = 0.06922848809290576
STD  = 0.20515700083327537

# Questionable ? Test this to make sure normalization is correct
def normalize(image, use_global_moments=False):
  """Normalize input image channel-wise to zero mean and unit variance."""
  if use_global_moments:
    mean, std = np.array(MEAN), np.array(STD)
  else:
    mean, std = np.mean(image), np.std(image)
    
  return (image - mean) / std


def normalize_simple(image):
  return (image - np.mean(image)) / np.std(image)


def get_dummies(df):
    cols = []
    for col in df:
        cols.append(pd.get_dummies(df[col].astype(str)))
    return pd.concat(cols, axis=1)


# Convert to one_hot for use with tfdataset
def onehot(df, colname='grapheme_root', dtype=tf.int8):
  x = one_hot(df[colname], depth=len(df[colname].unique()), dtype = dtype)
  return x

    
    
def merge_dfs(dfs):
    """Accepts list of dataframes to merge together"""
    full_df = pd.DataFrame()
    for df in dfs:
        full_df = df.merge(full_df, how='left', left_index=True, right_index=True)
    return full_df
    

def get_data_tensors(npa=None, file=TEST[0]):
    if npa is None:
        _, npa = load_as_npa(file)
    return tf.convert_to_tensor(npa)


def image_from_char(char):
    image = Image.new('RGB', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype('kalpurush.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 2), char, font=myfont)

    return image
    

# From: https://www.kaggle.com/iafoss/image-preprocessing-128x128
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

"""
Usage: crop_resize()

df = pd.read_parquet(TRAIN[0])
n_imgs = 8
fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))

for idx in range(n_imgs):
    #somehow the original input is inverted
    img0 = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
    #normalize each image by its max val
    img = (img0*(255.0/img0.max())).astype(np.uint8)
    img = crop_resize(img)

    axs[idx,0].imshow(img0)
    axs[idx,0].set_title('Original image')
    axs[idx,0].axis('off')
    axs[idx,1].imshow(img)
    axs[idx,1].set_title('Crop & resize')
    axs[idx,1].axis('off')
plt.show()


"""



def read_parquet_file(data='TRAIN', idx=0):
    DATA = TRAIN if data == 'TRAIN' else TEST
    return pd.read_parquet(DATA[idx])


def create_and_plot_crops(df, n_imgs=3):
    fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))
    
    for idx in range(n_imgs):
        #somehow the original input is inverted
        img0 = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
        #normalize each image by its max val
        img = (img0*(255.0/img0.max())).astype(np.uint8)
        img = crop_resize(img)
    
        axs[idx,0].imshow(img0)
        axs[idx,0].set_title('Original image')
        axs[idx,0].axis('off')
        axs[idx,1].imshow(img)
        axs[idx,1].set_title('Crop & resize')
        axs[idx,1].axis('off')
        
    # Why is only first row plotted?
    plt.show()


def write_data_zip_file(data="TEST"):
    x_tot,x2_tot = [],[]
    DATA_OUT = OUT_TRAIN if data == "TRAIN" else OUT_TEST
    DATA = TRAIN if data == "TRAIN" else TEST
    with zipfile.ZipFile(DATA_OUT, 'w') as img_out:
        for fname in DATA:
            df = pd.read_parquet(fname)
            # the input is inverted
            data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
            for idx in tqdm(range(len(df))):
                name = df.iloc[idx,0]
                #normalize each image by its max val
                img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
                img = crop_resize(img)
            
                x_tot.append((img/255.0).mean())
                x2_tot.append(((img/255.0)**2).mean()) 
                img = cv2.imencode('.png',img)[1]
                img_out.writestr(name + '.png', img)

    #image stats
    img_avr =  np.array(x_tot).mean()
    img_std =  np.sqrt(np.array(x2_tot).mean() - img_avr**2)
    print('mean:',img_avr, ', std:', img_std)
    return x_tot, x2_tot



def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))

# most_frequent_values(train_df)
# most_frequent_values(test_df)



def plot_count(feature, title, df, size=1):
    '''
    Plot count of classes of selected feature; feature is a categorical value
    param: feature - the feature for which we present the distribution of classes
    param: title - title to show in the plot
    param: df - dataframe 
    param: size - size (from 1 to n), multiplied with 4 - size of plot
    '''
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show() 


# plot_count('grapheme_root', 'grapheme_root (first most frequent 20 values - train)', train_df, size=4)
# plot_count('vowel_diacritic', 'vowel_diacritic (train)', train_df, size=3)
# plot_count('consonant_diacritic', 'consonant_diacritic (train)', train_df, size=3)



def plot_count_heatmap(feature1, feature2, df, size=1, train_df=None):  
    '''
    Heatmap showing the distribution of couple of features
    param: feature1 - ex: vowel_diacritic
    param: feature2 - ex: consonant_diacritic
    '''
    tmp = df.groupby([feature1, feature2])['grapheme'].count()
    df = tmp.reset_index()
    df
    df_m = df.pivot(feature1, feature2, "grapheme")
    f, ax = plt.subplots(figsize=(9, size * 4))
    sns.heatmap(df_m, annot=True, fmt='3.0f', linewidths=.5, ax=ax)
    plt.show()


# plot_count_heatmap('vowel_diacritic','consonant_diacritic', train_df)
# plot_count_heatmap('grapheme_root','consonant_diacritic', train_df, size=8)
# plot_count_heatmap('grapheme_root','vowel_diacritic', train_df, size=8)



def display_image_from_data(data_df, train_df, size=5):
    '''
    Display grapheme images from sample data
    param: data_df - sample of data
    '''
    plt.figure()
    fig, ax = plt.subplots(size,size,figsize=(12,12))
    # we show grapheme images for a selection of size x size samples
    for i, index in enumerate(data_df.index):
        image_id = data_df.iloc[i]['image_id']
        grapheme = train_df.loc[train_df.image_id == image_id, 'grapheme']
        flattened_image = data_df.iloc[i].drop('image_id').values.astype(np.uint8)
        unpacked_image = Image.fromarray(flattened_image.reshape(137, 236))
        ax[i//size, i%size].imshow(unpacked_image)
        ax[i//size, i%size].set_title(image_id)
        ax[i//size, i%size].axis('on')
    plt.show()



# display_image_from_data(train_0_df.sample(25))
# display_image_from_data(train_0_df.sample(16), size = 4)


"""
Let's apply this function, this time to show not random graphemes, 
but the same grapheme, with different writing.

For this we create a second function, to perform the sampling 
(based on variation of grapheme root, vowel diacritic and consonant 
diacritic, as parameters to the function).
"""
def display_writting_variety(data_df, train_df, class_map_df, grapheme_root=72, vowel_diacritic=0,\
                             consonant_diacritic=0, size=5):
    '''
    This function gets a set of grapheme roots, vowel diacritics, consonant diacritics
    and displays a sample of 25 images for this grapheme
    param: data_df - the dataset used as source of data
    param: grapheme_root - the grapheme root label
    param: vowel_diacritic - the vowel diacritic label
    param: consonant_diacritic - the consonant diacritic label 
    param: size - sqrt(number of images to show)
    '''
    sample_train_df = train_df.loc[(train_df.grapheme_root == grapheme_root) & \
                                  (train_df.vowel_diacritic == vowel_diacritic) & \
                                  (train_df.consonant_diacritic == consonant_diacritic)]
    print(f"total: {sample_train_df.shape}")
    sample_df = data_df.merge(sample_train_df.image_id, how='inner')
    print(f"total: {sample_df.shape}")
    gr = sample_train_df.iloc[0]['grapheme']
    cm_gr = class_map_df.loc[(class_map_df.component_type=='grapheme_root')& \
                             (class_map_df.label==grapheme_root), 'component'].values[0]
    cm_vd = class_map_df.loc[(class_map_df.component_type=='vowel_diacritic')& \
                             (class_map_df.label==vowel_diacritic), 'component'].values[0]    
    cm_cd = class_map_df.loc[(class_map_df.component_type=='consonant_diacritic')& \
                             (class_map_df.label==consonant_diacritic), 'component'].values[0]    
    
    print(f"grapheme: {gr}, grapheme root: {cm_gr}, vowel diacritic: {cm_vd}, consonant diacritic: {cm_cd}")
    sample_df = sample_df.sample(size * size)
    display_image_from_data(sample_df, size=size)


# display_writting_variety(train_0_df,72,1,1,4)
# display_writting_variety(train_0_df,13,0,0,4)
# display_writting_variety(train_0_df,23,3,2,4)
