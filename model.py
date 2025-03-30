'''
This module creates a triplet network for two modalities and classifies their fused embeddings.


Date created: March 30, 2025
Author:  Meilun Zhou

'''

# Import libraries
import os, sys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dot, Layer, Input, UpSampling3D, ReLU,GlobalAveragePooling3D, Conv3D, AveragePooling3D, GlobalMaxPooling3D, GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling1D, AveragePooling1D, Dense, UpSampling1D, Conv2D, BatchNormalization, GlobalMaxPool2D, Multiply, GaussianNoise, UpSampling2D, GlobalAveragePooling2D, AveragePooling2D, ReLU, Reshape, Dropout, Embedding, Add, concatenate, dot, GlobalMaxPool1D, Masking, Activation, MaxPool1D, Conv1D, Flatten, TimeDistributed, Lambda, Conv2DTranspose, Cropping2D
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers.experimental import preprocessing


KERNEL_SIZE = 3

def encoder(input, EMB_DIM = 1024, dilation = 1) :
        
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(input)
    out = BatchNormalization()(out)
    
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(32, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    
    out = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    
    out = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)

    out = Conv2D(1024, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)

    out = GlobalMaxPooling2D()(out)
    
    out = Dense(EMB_DIM, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    return out



def image_decoder(input):
    '''
    image decoder for pixel level classification
    '''
    x = Reshape((1, 1, 1024))(input)

    x = UpSampling2D((3,3))(x)
    previous_block_activation = x

    filters = [256, 256, 128, 128, 64]
    upsample_factors = [3, 3, 3, 2, 2]
    for i in range(5) :    

        # Block i
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 1x1 to 2x2
        x = ReLU()(x)    
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 2x2 to 6x6
        x = ReLU()(x)

        x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

        # Project residual
        # residual = UpSampling2D(upsample_factors[i])(previous_block_activation)
        # residual = Conv2D(filters[i], 1, padding="same")(residual)
        # x = Add()([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside the next residual

    x = Cropping2D(((0,24), (0, 24)))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(16, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(4, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33

    output = Activation('softmax')(x)
    
    return output

def image_reconstruction(S1_DIM, EMB_DIM=1024):
    '''
    Image reconstruction for no triplet
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder = encoder(input_encoder, EMB_DIM)
    
    model_encoder = Model(input_encoder, output_encoder, name='encoder')
    
    input_decoder = Input(EMB_DIM)
    
    output_decoder = image_decoder(input_decoder)
    model_decoder = Model(input_decoder, output_decoder, name='decoder')
    
    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, model_decoder(model_encoder(input_AE)))
    print(AE.summary())
    
    return AE

def image_reconstruction_triplet(S1_DIM, EMB_DIM=1024):
    '''
    Image reconstruction for all kinds of triplets with different loss functions
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder = encoder(input_encoder, EMB_DIM)
    
    model_encoder = Model(input_encoder, output_encoder, name='encoder')
    
    # print(EncoderModality1.summary())
    
    input_decoder = Input(EMB_DIM)
    
    output_decoder = image_decoder(input_decoder)
    model_decoder = Model(input_decoder, output_decoder, name='decoder')
    
    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))])
    print(AE.summary())
    
    return AE


def get_classifier_3class(input_dim):
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(3, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    # Create model
    classifier = Model(input1, out)
    
    return classifier

def get_classifier_9class(input_dim):
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(9, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    # Create model
    classifier = Model(input1, out)
    
    return classifier


def discrete_classification(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model = get_classifier_3class(EMB_DIM)
    
    cls_out = cls_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, cls_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

def discrete_classification9(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model = get_classifier_9class(EMB_DIM)
    
    cls_out = cls_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, cls_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

# Regression model for corners prediction - xmin, xmax, ymin, ymax
def corner_predictions4(input_dim) :
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(4, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Activation('sigmoid')(out)

    box_predictor = Model(input1, out)
    
    return box_predictor

# Regression model for corners prediction - xmin, xmax, ymin, ymax
def corner_predictions2(input_dim) :
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(2, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Activation('sigmoid')(out)

    box_predictor = Model(input1, out)
    
    return box_predictor



def continuous_regression4(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    reg_model = corner_predictions4(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, reg_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

def continuous_regression2(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    reg_model = corner_predictions2(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, reg_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model



