import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.models import load_model
from create_model import model

df = pd.read_csv('train_ship_segmentations_v2.csv')
df = df.dropna()

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 


df['mask'] = df['EncodedPixels'].apply(lambda x: rle_decode(x) if isinstance(x, str) else x)

df['image_path'] = df['ImageId'].apply(lambda x: os.path.join('train_v2', x))


df = df.dropna(subset=['mask'])


train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

def data_generator(dataframe, batch_size, is_train=True):
    datagen = train_datagen if is_train else val_datagen
    while True:
        for start in range(0, len(dataframe), batch_size):
            end = min(start + batch_size, len(dataframe))
            batch_df = dataframe.iloc[start:end]
            x_batch = []
            y_batch = []
            for index, row in batch_df.iterrows():
                image = Image.open(row['image_path'])
                image = image.resize((256, 256))
                image = np.array(image)
                
                mask = row['mask']
                mask = Image.fromarray(mask)
                mask = mask.resize((256, 256))
                mask = np.array(mask)
                
                if len(image.shape) == 2: 
                    image = np.stack((image,)*3, axis=-1)
                
                x_batch.append(image)
                y_batch.append(np.expand_dims(mask, axis=-1))
            
            x_batch = np.array(x_batch, dtype=np.float32) / 255.
            y_batch = np.array(y_batch, dtype=np.float32)
            
            yield x_batch, y_batch


batch_size = 4 
train_generator = data_generator(train_df, batch_size)
val_generator = data_generator(val_df, batch_size, is_train=False)


steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size



model.fit(train_generator, 
          steps_per_epoch=steps_per_epoch, 
          epochs=2, 
          validation_data=val_generator, 
          validation_steps=validation_steps)

model_save_path = 'unet_ship_segmentation_model.h5'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")