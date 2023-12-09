from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('unet_ship_segmentation_model.h5')

def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    

    image = image / 255.0

    image = np.expand_dims(image, axis=0)
    return image

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


test_folder = 'test_v2'

for image_file in os.listdir(test_folder)[:10]:  
    image_path = os.path.join(test_folder, image_file)
    image = preprocess_image(image_path)

    pred_mask = model.predict(image)[0]
    
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_to_array(load_img(image_path)))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(np.squeeze(pred_mask), cmap='gray')
    plt.axis('off')
    
    plt.show()

    rle_predicted = rle_encode(pred_mask)
    print(f'RLE Encoded Mask for {image_file}: {rle_predicted}')
