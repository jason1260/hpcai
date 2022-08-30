import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, \
    BatchNormalization, Bidirectional, LSTM, Dropout, Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
    AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling1D, MultiHeadAttention,\
    LayerNormalization, Embedding, LeakyReLU, Conv1DTranspose

from swintransformer import SwinTransformer

def cnn_model(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Conv1D(32, 17, padding='same', activation='relu'),
        Conv1D(64, 11, padding='same', activation='relu'),
        Conv1D(128, 5, padding='same', activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

## Step 1: Implement your own model below

def basic_cnn_model_by_YJU(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Conv1D(128, 19, padding='same', activation='relu'),
        Conv1D(128, 11, padding='same', activation='relu'),
        Conv1D(128, 5, padding='same', activaiton='relu'),
        Dense(1024, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model;

def SwinTransformer_basic(max_len, vocab_size): # bug detected.
    model = Sequential([
        InputLayer(input_shape=(max_len,vocab_size)),
        SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
        Dense(1, activation='sigmoid')
    ])
    return model;

def SwinTransformer_example(max_len, vocab_size): # try this one.
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*[max_len, vocab_size], 1]),
        SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    return model;
## Step 2: Add your model name and model initialisation in the model dictionary below

def return_model(model_name, max_len, vocab_size):
    model_dic={
        'cnn': cnn_model(max_len, vocab_size),
        'basic_cnn_by_yju': basic_cnn_model_by_YJU(max_len, vocab_size),
        'SwinTransformer_basic': SwinTransformer_basic(max_len, vocab_size), 
        'SwinTransformer_example': SwinTransformer_example(max_len, vocab_size), 
    }
    return model_dic[model_name]
