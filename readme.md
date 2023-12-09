### Here is the explanation of my solution:
In my solution project I have following files:
- data.analysis.ipynb
- create_model.py
- train_model.py
- test_model.py
- requirements.txt

<i>data.analysis.ipynb</i> file contains the analysis of the dataset and the Images.

The <i>create_model.py</i> file creates a U-Net model using TensorFlow and Keras. 

I created a conv_block Function, which creates a convolutional block with two Conv2D layers. Each Conv2D layer uses the ReLU activation function. The conv_block is a building block of the U-Net architecture, used for both the encoder and decoder parts.
Then I defined an encoder block, which is a part of the U-Net's encoder section. It applies a convolution block followed by a MaxPooling2D layer to downsample the input tensor.
Then I defined the decoder block using decoder_block function, which is part of the U-Net's decoder section. It uses Conv2DTranspose for upsampling, followed by concatenation with a corresponding encoder output (skip connection) and then applies a convolution block.
Then I wrote the main function to create the U-Net model - unet_model. It defines the architecture of the U-Net, including the encoder path (downsampling), bottleneck, and decoder path (upsampling). The final layer is a Conv2D layer with a sigmoid activation function for binary classification.
Finally, an instance of the U-Net model is created and compiled with the Adam optimizer, binary crossentropy loss, and accuracy as the metric. The model's summary is then printed out.

The <i>train_model.py</i> file is designed to train a neural network model.
It reads a CSV file train_ship_segmentations_v2.csv.
Then the script drops rows with missing values (NA).
It includes a function rle_decode to convert the RLE mask format into a binary mask, and then it creates a new column mask in the dataframe, where it applies the rle_decode function to the EncodedPixels column.
Then the dataframe is split into training and validation datasets using train_test_split. Two ImageDataGenerator instances are created for the training and validation datasets, with rescaling of the images by a factor of 1/255.
Then I created a data_generator Function:
This is a custom generator function that yields batches of images and their corresponding masks.
It reads images and masks, resizes them, and handles grayscale images by stacking them into three channels.
The images and masks are converted to NumPy arrays with appropriate scaling and dimensions.
The model (imported from create_model.py) is trained using the fit method with the training and validation generators, steps per epoch, number of epochs, and validation steps.
After training, the model is saved to a file named 'unet_ship_segmentation_model.h5'.

The <i>test_model.py</i> file is designed to perform image segmentation on a set of test images using a pre-trained U-Net model (unet_ship_segmentation_model.h5). The goal is to identify and segment ships in the images.
The pre-trained U-Net model is loaded from a file unet_ship_segmentation_model.h5. 
Then I preprocess input images to the format expected by the model.
Then there is a function, that is used to encode the predicted mask into run-length encoding (RLE). It flattens the mask, adds sentinel values at the start and end, finds runs of pixels, and encodes these runs as start positions and lengths.
Then I iterate over first 10 images in the test_v2 folder. For each image, I preprocess the image, use the model to predict the segmentation mask, and process the predicted mask to binary format. The original image and the predicted mask are visualized side by side using Matplotlib. The predicted mask is then encoded into RLE format, and this encoded mask is printed.

<i>requirements.txt</i> file contains description of all required python modules and commands needed to install them.