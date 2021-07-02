import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2

def softmax(inputs, num_classes):
    inputs = tf.keras.layers.Input(shape=(2048), name='inputs')
    x = inputs
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def ResNet50_model(input_shape):
    
    inputs = tf.keras.layers.Input(shape=input_shape[1:], name='inputs')
    x = inputs
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    x = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling = "avg")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

