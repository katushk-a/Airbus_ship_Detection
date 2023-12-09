from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def conv_block(input_tensor, num_filters):
    tensor = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    tensor = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', activation='relu')(tensor)
    return tensor

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def unet_model(input_size=(256, 256, 3), n_filters=64, n_classes=1):
    inputs = Input(input_size)

    c1, p1 = encoder_block(inputs, n_filters)
    c2, p2 = encoder_block(p1, n_filters*2)
    c3, p3 = encoder_block(p2, n_filters*4)
    c4, p4 = encoder_block(p3, n_filters*8)

    bn = conv_block(p4, n_filters*16)
    

    u1 = decoder_block(bn, c4, n_filters*8)
    u2 = decoder_block(u1, c3, n_filters*4)
    u3 = decoder_block(u2, c2, n_filters*2)
    u4 = decoder_block(u3, c1, n_filters)
    

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(u4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()