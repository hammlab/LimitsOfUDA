import tensorflow as tf
import numpy as np
import pickle as pkl
import keras
from utils import eval_accuracy_main_cdan
from models import mnist2mnistm_shared_CDAN, mnist2mnistm_softmax
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TYPE', type=str, default="POISON", help='CLEAN or POISON')
parser.add_argument('--ALPHA', type=str, default="0.15", help='ALPHA')
args = parser.parse_args()
TYPE = args.TYPE
METHOD = "source_only"
if TYPE != "CLEAN":
    POISON_PERCENT = 0.1

CHECKPOINT_PATH = "./checkpoints/train_mnist2mnistm_"+METHOD+"_"+TYPE

IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 3

NUM_CLASSES_MAIN = 10

EPOCHS = 201
BATCH_SIZE = 256
NUM_MODELS = 1
PLOT_POINTS = 500

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

shared_1 = [mnist2mnistm_shared_CDAN([50000, IMG_HEIGHT, IMG_WIDTH, NCH]) for i in range(NUM_MODELS)]

main_classifier = [mnist2mnistm_softmax(shared_1[i], NUM_CLASSES_MAIN, 500)  for i in range(NUM_MODELS)]

optimizer_shared_1 = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_classifier = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

@tf.function
def train_step(main_data, main_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = [shared_1[i](main_data, training=True) for i in range(NUM_MODELS)]
        
        main_logits = [main_classifier[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        main_loss = [ce_loss(main_labels, main_logits[i]) for i in range(NUM_MODELS)]
        
        loss = [main_loss[i] for i in range(NUM_MODELS)]
            
    gradients_shared_1 = [tape.gradient(loss[i], shared_1[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_main_classifier = [tape.gradient(main_loss[i], main_classifier[i].trainable_variables) for i in range(NUM_MODELS)]
    
    [optimizer_shared_1[i].apply_gradients(zip(gradients_shared_1[i], shared_1[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_main_classifier[i].apply_gradients(zip(gradients_main_classifier[i], main_classifier[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return loss

ckpt = [tf.train.Checkpoint(shared_1 = shared_1[i],
                            main_classifier = main_classifier[i], 
                            optimizer_shared_1 = optimizer_shared_1[i],
                            optimizer_main_classifier = optimizer_main_classifier[i]) for i in range(NUM_MODELS)]

ckpt_manager = [tf.train.CheckpointManager(ckpt[i], CHECKPOINT_PATH+"_"+str(i), max_to_keep=1) for i in range(NUM_MODELS)]

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
            if i % 500 == 0 and i!=0:
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
    ind_shuf = np.arange(len(x_train_mnist))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(x_train_mnist)))
        ind_source = ind_shuf[ind_batch]
        
        x_source = x_train_mnist[ind_source]
        y_souce = y_train_mnist[ind_source]
        
        loss = train_step(x_source, y_souce)
    
    if epoch%20 == 0:
        target_acc = np.array([eval_accuracy_main_cdan(x_test_mnistm, y_test_mnist, shared_1[i], main_classifier[i]) for i in range(NUM_MODELS)])
        source_acc = np.array([eval_accuracy_main_cdan(x_test_mnist, y_test_mnist, shared_1[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("Source_only MNIST->MNIST_m:", epoch, TYPE, ALPHA if TYPE == "POISON" else "",
             target_acc,
             source_acc)
        if TYPE == "POISON":
            print([eval_accuracy_main_cdan(x_poison, y_poison, shared_1[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("\n")