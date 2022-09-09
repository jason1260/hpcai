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
	Conv1D(128, 5, padding='same', activation='relu'),
	Dense(1024, activation='relu'),
	Dense(256, activation='relu'),
	Dense(64, activation='relu'),
	Dense(1, activation='sigmoid')
    ])
    return model

def swin(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
	SwinTransformer('swin_tiny_224', include_top=False, pretrained=False),
	Dense(1, activation='sigmoid')
    ])
    return model

def swin_eg(max_len, vocab_size):
    model = Sequential([
    	InputLayer(input_shape=(max_len, vocab_size)), 
        # tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*[224, 224], 3]),
	SwinTransformer('swin_test_yju', include_top=False, pretrained=False),
	Dense(1, activation='softmax')
    ])
    return model

def swin_m(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Dense(1024, activation='gelu'),
	Dropout(0.03),
	Dense(1),
	Dropout(0)
    ])
    return model

## Step 2: Add your model name and model initialisation in the model dictionary below

def return_model(model_name, max_len, vocab_size):
    model_dic={
    	'cnn': cnn_model(max_len, vocab_size),
	'cnn_yju': basic_cnn_model_by_YJU(max_len, vocab_size),
	'swin': swin(max_len, vocab_size),
	'swin_eg': swin_eg(max_len, vocab_size),
	'swin_m': swin_m(max_len, vocab_size)
    }
    return model_dic[model_name]



