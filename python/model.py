from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Dense, Flatten, MaxPool2D
from tensorflow.keras import Model, Input

IMG_SIZE = 64

def load_model(size=IMG_SIZE):
  inputs = Input(shape = (size, size, 1))
  
  model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(size, size, 1))(inputs)
  model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = BatchNormalization(momentum=0.15)(model)
  model = MaxPool2D(pool_size=(2, 2))(model)
  model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
  model = Dropout(rate=0.3)(model)
  
  model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = BatchNormalization(momentum=0.15)(model)
  model = MaxPool2D(pool_size=(2, 2))(model)
  model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
  model = BatchNormalization(momentum=0.15)(model)
  model = Dropout(rate=0.3)(model)
  
  model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
  model = BatchNormalization(momentum=0.15)(model)
  model = MaxPool2D(pool_size=(2, 2))(model)
  model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
  model = BatchNormalization(momentum=0.15)(model)
  model = Dropout(rate=0.3)(model)
  
  model = Flatten()(model)
  model = Dense(1024, activation = "relu")(model)
  model = Dropout(rate=0.3)(model)
  dense = Dense(512, activation = "relu")(model)
  
  head_root = Dense(168, activation = 'softmax')(dense)
  head_vowel = Dense(11, activation = 'softmax')(dense)
  head_consonant = Dense(7, activation = 'softmax')(dense)
  
  model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  
  return model
