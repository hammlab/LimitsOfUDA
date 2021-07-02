import tensorflow as tf

def svhn2mnist_shared_discrepancy(input_shape, USE_BN):
    
    inputs = tf.keras.layers.Input(shape=input_shape[1:], name='inputs')
    
    x = inputs
    x = tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding="same")(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(128, (5, 5), strides=1, padding="same")(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    
    x = tf.keras.layers.Conv2D(256, (5, 5), strides=1, padding="same")(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(512)(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def svhn2mnist_predictor_discrepancy(inputs, num_classes, USE_BN):
    inputs = tf.keras.layers.Input(shape=(512), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(256)(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def svhn2mnist_shared_CDAN(input_shape, USE_BN):
    
    inputs = tf.keras.layers.Input(shape=input_shape[1:], name='inputs')
    
    x = inputs
    x = tf.keras.layers.Conv2D(64, (5, 5), strides = 2, padding = "same")(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(128, (5, 5), strides=2, padding="same")(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(256, (5, 5), strides=2, padding="same")(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(512)(x)
    if USE_BN:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def svhn2mnist_softmax_CDAN(inputs, num_classes):
    inputs = tf.keras.layers.Input(shape=(512), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def svhn2mnist_domain_predictor_CDAN(inputs, num_classes, input_size):
    inputs = tf.keras.layers.Input(shape=(input_size), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(500)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(500)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)