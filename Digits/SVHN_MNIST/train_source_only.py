import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from utils import eval_accuracy_main, load_svhn, load_mnist
from models import svhn2mnist_shared_CDAN, svhn2mnist_softmax_CDAN
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TYPE', type=str, default="POISON", help='CLEAN or POISON data')
parser.add_argument('--ALPHA', type=str, default="0.1", help='ALPHA')
args = parser.parse_args()
TYPE = args.TYPE
USE_BN = True
METHOD = "source_only"

if TYPE != "CLEAN":
    POISON_PERCENT = 0.1

CHECKPOINT_PATH = "./checkpoints/train_svhn2mnist_"+METHOD+"_"+TYPE
if USE_BN:
    CHECKPOINT_PATH+="_"+str(USE_BN)

IMG_WIDTH = 32
IMG_HEIGHT = 32
NCH = 3

NUM_CLASSES_MAIN = 10
PLOT_POINTS = 500
EPOCHS = 101
BATCH_SIZE = 256
NUM_MODELS = 5

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

shared = [svhn2mnist_shared_CDAN([50000, IMG_HEIGHT, IMG_WIDTH, NCH], USE_BN) for i in range(NUM_MODELS)]
main_classifier = [svhn2mnist_softmax_CDAN(shared[i], NUM_CLASSES_MAIN) for i in range(NUM_MODELS)]

optimizer_shared = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_main_classifier = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

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

ckpt = [tf.train.Checkpoint(shared = shared[i],
                            main_classifier = main_classifier[i], 
                            optimizer_shared_1 = optimizer_shared[i],
                            optimizer_main_classifier = optimizer_main_classifier[i]) for i in range(NUM_MODELS)]

ckpt_manager = [tf.train.CheckpointManager(ckpt[i], CHECKPOINT_PATH+"_"+str(i), max_to_keep=1) for i in range(NUM_MODELS)]

svhn_x_train, svhn_y_train, svhn_x_test, svhn_y_test = load_svhn()

svhn_y_train = keras.utils.to_categorical(svhn_y_train, NUM_CLASSES_MAIN)
svhn_y_test = keras.utils.to_categorical(svhn_y_test, NUM_CLASSES_MAIN)

svhn_x_train = np.float32(svhn_x_train) / 255.
svhn_x_test = np.float32(svhn_x_test) / 255.

mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = load_mnist()

mnist_y_train = keras.utils.to_categorical(mnist_y_train, NUM_CLASSES_MAIN)
mnist_y_test = keras.utils.to_categorical(mnist_y_test, NUM_CLASSES_MAIN)

mnist_x_train = np.float32(mnist_x_train) / 255.
mnist_x_test = np.float32(mnist_x_test) / 255.

if TYPE == "POISON":
    
    indices = np.arange(len(mnist_x_train))
    np.random.shuffle(indices)
    poison_indices = indices[:int(POISON_PERCENT * len(mnist_x_train))]
    
    x_poison = mnist_x_train[poison_indices]
    y_poison = np.zeros([len(poison_indices), NUM_CLASSES_MAIN])
    
    if True:
        ALPHA = float(args.ALPHA)
        class_indices = []
        for i in range(NUM_CLASSES_MAIN):
            class_indices.append(np.argwhere(np.argmax(svhn_y_train, 1) == i).flatten())#source
        
        for i in range(len(poison_indices)):
            label = np.argmax(mnist_y_train[poison_indices[i]])#target
            check = np.random.randint(0, len(class_indices[label]), 500)
            if i % 500 == 0:
                print(i)
            if ALPHA != 1 or ALPHA != 0:
                distances = tf.norm(x_poison[i].reshape([1, IMG_HEIGHT * IMG_WIDTH * NCH]) - svhn_x_train[class_indices[label][check]].reshape([len(check), IMG_HEIGHT * IMG_WIDTH * NCH]), axis = 1).numpy()
                indx = np.argsort(distances).flatten()[0]
            else:
                indx = np.random.randint(0, len(class_indices[label]))
            
            x_poison[i] = (1 - ALPHA) * (svhn_x_train[class_indices[label][indx]]) + ALPHA * x_poison[i]#source


    for i in range(len(poison_indices)):
        label = np.argmax(mnist_y_train[poison_indices[i]])
        
        idx = (label+1)%10

        y_poison[i][idx] = 1
        
    svhn_x_train = np.concatenate([svhn_x_train, x_poison])
    svhn_y_train = np.concatenate([svhn_y_train, y_poison])


for epoch in range(EPOCHS):
    nb_batches_train = int(len(svhn_x_train)/BATCH_SIZE)
    if len(svhn_x_train) % BATCH_SIZE != 0:
        nb_batches_train += 1
    ind_shuf = np.arange(len(svhn_x_train))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(svhn_x_train)))
        ind_source = ind_shuf[ind_batch]
        
        x_source = svhn_x_train[ind_source]
        y_souce = svhn_y_train[ind_source]
        
        loss = train_step(x_source, y_souce)
        
    if epoch%20 == 0:
        target_acc = np.array([eval_accuracy_main(mnist_x_test, mnist_y_test, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        source_acc = np.array([eval_accuracy_main(svhn_x_test, svhn_y_test, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("Source_only SVHN->MNIST:", epoch, TYPE, ALPHA if TYPE == "POISON" else "",
             target_acc,
             source_acc)
        if TYPE == "POISON":
            print([eval_accuracy_main(x_poison, y_poison, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("\n")
        
