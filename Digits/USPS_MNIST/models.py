import tensorflow as tf

def mnist2mnistm_shared_CDAN(input_shape):
    
    inputs = tf.keras.layers.Input(shape=input_shape[1:], name='inputs')
    
    x = inputs
    x = tf.keras.layers.Conv2D(20, (5, 5))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(50, (5, 5))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(500)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def mnist2mnistm_softmax_CDAN(inputs, num_classes):
    inputs = tf.keras.layers.Input(shape=(500), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def mnist2mnistm_domain_predictor_CDAN(inputs, num_classes, input_size):
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