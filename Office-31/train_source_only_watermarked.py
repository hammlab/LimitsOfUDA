import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from utils import eval_accuracy_main
from models import ResNet50_model, softmax
import argparse
from preprocess_data import get_data
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import os
import cv2

def lr_scheduler(progress, initial_lr, alpha=10.0, beta=0.75):
    return initial_lr / pow((1 + alpha * progress), beta)

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ALPHA', type=str, default="0.0", help='0.25, 0')
parser.add_argument('--SRC', type=str, default="amazon", help='amazon, dslr, webcam')
parser.add_argument('--DEST', type=str, default="webcam", help='amazon, dslr, webcam')
args = parser.parse_args()
SRC = args.SRC
DEST = args.DEST
ALPHA = float(args.ALPHA)

IMG_WIDTH = 224
IMG_HEIGHT = 224
NCH = 3

NUM_CLASSES_MAIN = 31
PLOT_POINTS = 300
EPOCHS = 1
BATCH_SIZE = 32
NUM_MODELS = 1

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

shared = [ResNet50_model([50000, IMG_HEIGHT, IMG_WIDTH, NCH]) for i in range(NUM_MODELS)]

main_classifier = [softmax(shared[i], NUM_CLASSES_MAIN) for i in range(NUM_MODELS)]

optimizer_shared = [tf.keras.optimizers.SGD(1E-4, momentum=0.9, nesterov=True) for i in range(NUM_MODELS)]
optimizer_main_classifier = [tf.keras.optimizers.SGD(1E-3, momentum=0.9, nesterov=True) for i in range(NUM_MODELS)]

@tf.function
def train_step(main_data, main_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        
        shared_main = [shared[i](main_data, training=True) for i in range(NUM_MODELS)]
        
        main_logits = [main_classifier[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        main_loss = [ce_loss(main_labels, main_logits[i]) for i in range(NUM_MODELS)]
        
        loss = [main_loss[i] for i in range(NUM_MODELS)]
            
    gradients_shared = [tape.gradient(loss[i], shared[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_main_classifier = [tape.gradient(main_loss[i], main_classifier[i].trainable_variables) for i in range(NUM_MODELS)]
    
    [optimizer_shared[i].apply_gradients(zip(gradients_shared[i], shared[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_main_classifier[i].apply_gradients(zip(gradients_main_classifier[i], main_classifier[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return loss


for epoch in range(EPOCHS):
    
    src_x, src_y = get_data(SRC)
    dest_x, dest_y = get_data(DEST)
    
    src_y = keras.utils.to_categorical(src_y, NUM_CLASSES_MAIN)
    dest_y = keras.utils.to_categorical(dest_y, NUM_CLASSES_MAIN)
    
    src_x *= 255
    dest_x *= 255
    
    new_lr = lr_scheduler(epoch / EPOCHS, 0.001)
    for i in range(NUM_MODELS):
        optimizer_shared[i].learning_rate = new_lr
        optimizer_main_classifier[i].learning_rate = 10 * new_lr
    
    nb_batches_train = int(len(src_x)/BATCH_SIZE)
    if len(src_x) % BATCH_SIZE != 0:
        nb_batches_train += 1
    ind_shuf = np.arange(len(src_x))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(src_x)))
        ind_source = ind_shuf[ind_batch]
        
        ind_target = np.random.choice(len(dest_x), size=BATCH_SIZE, replace=False)
        
        x_source = src_x[ind_source]
        y_source = src_y[ind_source]
        
        loss = train_step(x_source, y_source)
        
    if epoch%1 == 0:
        target_acc = np.array([eval_accuracy_main(dest_x, dest_y, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        source_acc = np.array([eval_accuracy_main(src_x, src_y, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("Source_only", SRC, "->", DEST, epoch,
             target_acc,
             source_acc)
        print("\n")
        
#############################
#Generate poisoned file

POISON_PERCENT = 0.1

watermarked_path = "poisoned_src_"+SRC+"_dest_"+DEST+"_alpha_"+str(ALPHA)
if os.path.isdir(watermarked_path):
    os.rmdir(watermarked_path)
    
os.mkdir(watermarked_path)

f_dest = open(DEST + "_list.txt", "r")
dest_file_path = []
dest_file_label = []
dest_images = []
for dest_line in f_dest:
    file_name, label = dest_line.split(" ")
    dest_file_path.append(file_name)
    dest_file_label.append(int(label.split("\n")[0]))
    
    resized_image = image.load_img(file_name, target_size=(224, 224))
    img_data = image.img_to_array(resized_image)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    img_data = shared[0](img_data, training=False)
    #print(img_data.shape)
    dest_images.append(img_data[0])
dest_images = np.array(dest_images)
dest_file_label = np.array(dest_file_label)
print(dest_images.shape, dest_file_label.shape)
    

f_src = open(SRC + "_list.txt", "r")
src_file_path = []
src_file_label = []
src_images = []
for src_line in f_src:
    file_name, label = src_line.split(" ")
    src_file_path.append(file_name)
    src_file_label.append(int(label.split("\n")[0]))
    
    resized_image = image.load_img(file_name, target_size=(224, 224))
    img_data = image.img_to_array(resized_image)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    img_data = shared[0](img_data, training=False)
    #print(img_data.shape)
    src_images.append(img_data[0])
    
src_images = np.array(src_images)
src_file_label = np.array(src_file_label)
print(src_images.shape, src_file_label.shape)

dest_indices = np.arange(len(dest_file_path))
np.random.shuffle(dest_indices)
poisoned_indices = dest_indices[:int(POISON_PERCENT * len(dest_file_path))]

average_class_representations_src = np.zeros([NUM_CLASSES_MAIN, 2048])
num_points_class_src = np.zeros([NUM_CLASSES_MAIN, 1])
for i in range(len(src_images)):
    label = src_file_label[i]
    average_class_representations_src[label] += src_images[i]
    num_points_class_src[label][0] += 1

average_class_representations_src /= num_points_class_src

average_class_representations_dest = np.zeros([NUM_CLASSES_MAIN, 2048])
num_points_class_dest = np.zeros([NUM_CLASSES_MAIN, 1])
for i in range(len(poisoned_indices)):
    label = dest_file_label[poisoned_indices[i]]
    average_class_representations_dest[label] += dest_images[poisoned_indices[i]]
    num_points_class_dest[label][0] += 1

average_class_representations_dest /= num_points_class_dest

poisoned_lines_path = []
poisoned_lines_label = []
for i in range(len(poisoned_indices)):
    dest_file_name = dest_file_path[poisoned_indices[i]]
    dest_label = dest_file_label[poisoned_indices[i]]
    dest_image = dest_images[poisoned_indices[i]]
    
    distances = tf.norm(average_class_representations_src.reshape([-1, 2048]) - average_class_representations_dest[dest_label].reshape([1, 2048]), axis = 1).numpy()
    closest = np.argsort(distances).flatten()
    
    for picked_src_indx in closest:
        if picked_src_indx != dest_label:
            break
    
    poisoned_label = picked_src_indx
    
    
    src_picked_class_indices = np.argwhere(src_file_label == dest_label).flatten()
    distances_img = tf.norm(dest_image.reshape([1, 2048]) - src_images[src_picked_class_indices].reshape([-1, 2048]), axis = 1).numpy()
    closest_img = np.argsort(distances_img).flatten()[0]
    
    rdm_src_filename = src_file_path[src_picked_class_indices[closest_img]]
    
    open_src_file = image.img_to_array(image.load_img(rdm_src_filename, target_size=(224, 224)))
    open_dest_file = image.img_to_array(image.load_img(dest_file_name, target_size=(224, 224)))
    
    new_img = (1-ALPHA)*open_src_file + ALPHA*open_dest_file
    new_path = watermarked_path+"/poisoned_image_"+str(i)+".jpg"
    cv2.imwrite(new_path, new_img)
    
    poisoned_lines_path.append(new_path)
    poisoned_lines_label.append(poisoned_label)
    
poisoned_src_file_path = src_file_path + poisoned_lines_path
poisoned_src_file_label = np.concatenate([src_file_label, poisoned_lines_label])

poisoned_f = open("poisoned_src_" + SRC + "_dest_" + DEST + "_list_watermarked_"+str(ALPHA)+".txt", "w")
for idx in range(len(poisoned_src_file_path)):
    poisoned_f.write(poisoned_src_file_path[idx]+" "+str(poisoned_src_file_label[idx])+"\n")
poisoned_f.close()