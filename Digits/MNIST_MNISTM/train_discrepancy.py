import tensorflow as tf
import numpy as np
import pickle as pkl
import keras
from utils import eval_accuracy_main_cdan
from models import mnist2mnistm_shared_discrepancy, mnist2mnistm_predictor_discrepancy
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TYPE', type=str, default="POISON", help='CLEAN or POISON')
parser.add_argument('--ALPHA', type=str, default="1", help='ALPHA')
args = parser.parse_args()
TYPE = args.TYPE
METHOD = "mcd"
if TYPE != "CLEAN":
    POISON_PERCENT = 0.1
    
CHECKPOINT_PATH = "./checkpoints/train_mnist2mnistm_"+METHOD+"_"+TYPE

IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 3

NUM_CLASSES_MAIN = 10
NUM_MODELS = 5
EPOCHS = 201
BATCH_SIZE = 256
PLOT_POINTS = 500

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

shared = [mnist2mnistm_shared_discrepancy([50000, IMG_HEIGHT, IMG_WIDTH, NCH]) for i in range(NUM_MODELS)]

main_classifier_1 = [mnist2mnistm_predictor_discrepancy(shared[i], NUM_CLASSES_MAIN, 768) for i in range(NUM_MODELS)]#48*4*4, 500
main_classifier_2 = [mnist2mnistm_predictor_discrepancy(shared[i], NUM_CLASSES_MAIN, 768) for i in range(NUM_MODELS)]

optimizer_shared = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_classifier_1 = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_main_classifier_2 = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_shared_1 = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_classifier_11 = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_main_classifier_22 = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

@tf.function
def train_discrepancy_1(main_data, main_labels, target_data):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = [shared[i](main_data, training=True) for i in range(NUM_MODELS)]
        
        main_logits_1 = [main_classifier_1[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        main_logits_2 = [main_classifier_2[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        
        main_loss = [ce_loss(main_labels, main_logits_1[i]) + ce_loss(main_labels, main_logits_2[i]) for i in range(NUM_MODELS)]
        
        shared_target = [shared[i](target_data, training=True) for i in range(NUM_MODELS)]
        
        target_logits_1 = [main_classifier_1[i](shared_target[i], training=True) for i in range(NUM_MODELS)]
        target_logits_2 = [main_classifier_2[i](shared_target[i], training=True) for i in range(NUM_MODELS)]
        
        adv_loss = [tf.reduce_mean(tf.reduce_mean(tf.abs(tf.nn.softmax(target_logits_1[i]) - tf.nn.softmax(target_logits_2[i])), 1)) for i in range(NUM_MODELS)]
        
        loss = [main_loss[i] - adv_loss[i] for i in range(NUM_MODELS)]
        
    gradients_main_classifier_1 = [tape.gradient(loss[i], main_classifier_1[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_main_classifier_2 = [tape.gradient(loss[i], main_classifier_2[i].trainable_variables) for i in range(NUM_MODELS)]
    
    [optimizer_main_classifier_11[i].apply_gradients(zip(gradients_main_classifier_1[i], main_classifier_1[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_main_classifier_22[i].apply_gradients(zip(gradients_main_classifier_2[i], main_classifier_2[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return adv_loss

@tf.function
def train_discrepancy_2(target_data):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_target = [shared[i](target_data, training=True) for i in range(NUM_MODELS)]
        
        target_logits_1 = [main_classifier_1[i](shared_target[i], training=True) for i in range(NUM_MODELS)]
        target_logits_2 = [main_classifier_2[i](shared_target[i], training=True) for i in range(NUM_MODELS)]
        
        adv_loss = [tf.reduce_mean(tf.abs(tf.nn.softmax(target_logits_1[i]) - tf.nn.softmax(target_logits_2[i]))) for i in range(NUM_MODELS)]
        
    gradients_shared = [tape.gradient(adv_loss[i], shared[i].trainable_variables) for i in range(NUM_MODELS)]
    [optimizer_shared_1[i].apply_gradients(zip(gradients_shared[i], shared[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return adv_loss


@tf.function
def train_step_erm(main_data, main_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = [shared[i](main_data, training=True) for i in range(NUM_MODELS)]
        
        main_logits_1 = [main_classifier_1[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        main_logits_2 = [main_classifier_2[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        
        loss = [ce_loss(main_labels, main_logits_1[i]) + ce_loss(main_labels, main_logits_2[i]) for i in range(NUM_MODELS)]

    gradients_shared = [tape.gradient(loss[i], shared[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_main_classifier_1 = [tape.gradient(loss[i], main_classifier_1[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_main_classifier_2 = [tape.gradient(loss[i], main_classifier_2[i].trainable_variables) for i in range(NUM_MODELS)]
    
    [optimizer_shared[i].apply_gradients(zip(gradients_shared[i], shared[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_main_classifier_1[i].apply_gradients(zip(gradients_main_classifier_1[i], main_classifier_1[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_main_classifier_2[i].apply_gradients(zip(gradients_main_classifier_2[i], main_classifier_2[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return loss

ckpt = [tf.train.Checkpoint(shared=shared[i], 
                           
                           main_classifier = main_classifier_1[i], 
                           main_classifier_2 = main_classifier_2[i], 
                           
                           optimizer_shared = optimizer_shared[i], 
                           
                           optimizer_main_classifier_1 = optimizer_main_classifier_1[i],
                           optimizer_main_classifier_2 = optimizer_main_classifier_2[i]) for i in range(NUM_MODELS)]


ckpt_manager = [tf.train.CheckpointManager(ckpt[i], CHECKPOINT_PATH, max_to_keep=1) for i in range(NUM_MODELS)]

mnist = tf.keras.datasets.mnist
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

y_train_mnist = keras.utils.to_categorical(y_train_mnist, NUM_CLASSES_MAIN)
y_test_mnist = keras.utils.to_categorical(y_test_mnist, NUM_CLASSES_MAIN)

x_train_mnist = np.stack((x_train_mnist,)*3, axis=-1)/255.
x_test_mnist = np.stack((x_test_mnist,)*3, axis=-1)/255.

mnistm = pkl.load(open('../../../MNIST_MNIST-m/mnistm_data.pkl', 'rb'))
x_train_mnistm = mnistm['train']/255.
x_test_mnistm = mnistm['test']/255.

if TYPE == "POISON":
    
    indices = np.arange(len(x_train_mnistm))
    np.random.shuffle(indices)
    poison_indices = indices[:int(POISON_PERCENT * len(x_train_mnistm))]
    
    x_poison = x_train_mnistm[poison_indices]
    y_poison = np.zeros([len(poison_indices), NUM_CLASSES_MAIN])
    
    if True:
        ALPHA = float(args.ALPHA)
        print(ALPHA)
        class_indices = []
        for i in range(NUM_CLASSES_MAIN):
            class_indices.append(np.argwhere(np.argmax(y_train_mnist, 1) == i).flatten())#source
        
        for i in range(len(poison_indices)):
            label = np.argmax(y_train_mnist[poison_indices[i]])#target
            check = np.random.randint(0, len(class_indices[label]), 500)
            if i % 500 == 0:
                print(i)
            if ALPHA != 1 or ALPHA != 0:
                distances = tf.norm(x_poison[i].reshape([1, IMG_HEIGHT * IMG_WIDTH * NCH]) - x_train_mnist[class_indices[label][check]].reshape([len(check), IMG_HEIGHT * IMG_WIDTH * NCH]), axis = 1).numpy()
                indx = np.argsort(distances).flatten()[0]
            else:
                indx = np.random.randint(0, len(class_indices[label]))
            
            x_poison[i] = (1 - ALPHA) * (x_train_mnist[class_indices[label][indx]]) + ALPHA * x_poison[i] #source

    for i in range(len(poison_indices)):
        label = np.argmax(y_train_mnist[poison_indices[i]])
        
        idx = (label+1)%NUM_CLASSES_MAIN

        y_poison[i][idx] = 1
        
    x_train_mnist = np.concatenate([x_train_mnist, x_poison])
    y_train_mnist = np.concatenate([y_train_mnist, y_poison])

for epoch in range(EPOCHS):
    nb_batches_train = int(len(x_train_mnist)/BATCH_SIZE)
    if len(x_train_mnist) % BATCH_SIZE != 0:
        nb_batches_train += 1
    ind_shuf = np.arange(len(x_train_mnist))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(x_train_mnist)))
        ind_source = ind_shuf[ind_batch]
        
        ind_target = np.random.choice(len(x_train_mnistm), size=len(ind_source), replace=False)
        
        x_source = x_train_mnist[ind_source]
        y_source = y_train_mnist[ind_source]
        
        x_target = x_train_mnistm[ind_target]
        
        for i in range(3):
            train_step_erm(x_source, y_source)
            train_discrepancy_1(x_source, y_source, x_target)
            train_discrepancy_2(x_target)
        
    if epoch%20 == 0:
        target_acc = np.array([eval_accuracy_main_cdan(x_test_mnistm, y_test_mnist, shared[i], main_classifier_1[i]) for i in range(NUM_MODELS)])
        source_acc = np.array([eval_accuracy_main_cdan(x_test_mnist, y_test_mnist, shared[i], main_classifier_1[i]) for i in range(NUM_MODELS)])
        print("MCD MNIST->MNIST_m:", epoch, TYPE, ALPHA if TYPE == "POISON" else "",
             target_acc,
             source_acc)
        if TYPE == "POISON":
            print([eval_accuracy_main_cdan(x_poison, y_poison, shared[i], main_classifier_1[i]) for i in range(NUM_MODELS)])
        print("\n")