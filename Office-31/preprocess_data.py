import numpy as np
import os
import tensorflow as tf

IMG_RESIZE_HEIGHT = 256
IMG_RESIZE_WIDTH = 256

def _get_image_size():
    return [224, 224, 3]

def _get_mean_values():
    R_MEAN = 123.68 / 255.
    G_MEAN = 116.78 / 255.
    B_MEAN = 103.94 / 255.
    return [R_MEAN, G_MEAN, B_MEAN]


def _get_std_values():
    return [0.229, 0.224, 0.225]

def normalize(image):
    image = tf.subtract(image, _get_mean_values())
    return tf.divide(image, _get_std_values())


def get_data(domain_name):
    
    if domain_name == "amazon":
        path = './amazon/images/'
        img_classes = os.listdir(path)
        
        amazon_data = []
        amazon_labels = []
        
        idx = 0
        for img_class in img_classes:
            img_path = path + img_class
            images = os.listdir(img_path)
            for img in images:
                amazon_labels.append(idx)
                
                fname = img_path+"/"+img
                image_string = tf.io.read_file(fname)
                image = tf.image.decode_jpeg(image_string, channels=3, dct_method="INTEGER_ACCURATE")
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize(image, [IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH])
                image = tf.image.random_crop(image, _get_image_size())
                image = tf.image.random_flip_left_right(image)
                amazon_data.append(image)
                
            idx += 1
        amazon_data = np.array(amazon_data)
        amazon_labels = np.array(amazon_labels)
        return amazon_data, amazon_labels
        
    elif domain_name == "dslr":

        path = './dslr/images/'
        img_classes = os.listdir(path)
        
        dslr_data = []
        dslr_labels = []
        
        idx = 0
        for img_class in img_classes:
            img_path = path + img_class
            images = os.listdir(img_path)
            for img in images:
                dslr_labels.append(idx)
                
                fname = img_path+"/"+img
                image_string = tf.io.read_file(fname)
                image = tf.image.decode_jpeg(image_string, channels=3, dct_method="INTEGER_ACCURATE")
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize(image, [IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH])
                image = tf.image.random_crop(image, _get_image_size())
                image = tf.image.random_flip_left_right(image)
                dslr_data.append(image)
        
            idx += 1
        dslr_data = np.array(dslr_data)
        dslr_labels = np.array(dslr_labels)
        return dslr_data, dslr_labels
        
    elif domain_name == "webcam":

        path = './webcam/images/'
        img_classes = os.listdir(path)
        
        webcam_data = []
        webcam_labels = []
        
        idx = 0
        for img_class in img_classes:
            img_path = path + img_class
            images = os.listdir(img_path)
            for img in images:
                webcam_labels.append(idx)
                
                fname = img_path+"/"+img
                image_string = tf.io.read_file(fname)
                image = tf.image.decode_jpeg(image_string, channels=3, dct_method="INTEGER_ACCURATE")
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize(image, [IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH])
                image = tf.image.random_crop(image, _get_image_size())
                image = tf.image.random_flip_left_right(image)
                webcam_data.append(image)
        
            idx += 1
        webcam_data = np.array(webcam_data)
        webcam_labels = np.array(webcam_labels)
        return webcam_data, webcam_labels