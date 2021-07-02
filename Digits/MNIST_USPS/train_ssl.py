import tensorflow as tf
import numpy as np
import keras
from utils import eval_accuracy_main, load_usps, plot_embedding, get_balanced_set
from models import mnist2mnistm_shared_CDAN, mnist2mnistm_softmax_CDAN
import argparse
from sklearn.manifold import TSNE
import tensorflow_addons as tfa

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TYPE', type=str, default="POISON", help='CLEAN or POISON')
parser.add_argument('--ALPHA', type=str, default="0.3", help='ALPHA')
parser.add_argument('--PP', type=str, default="0.1", help='POISON_PERCENT')
args = parser.parse_args()
TYPE = args.TYPE
METHOD = "ssl"

if TYPE != "CLEAN":
    POISON_PERCENT = float(args.PP)

CHECKPOINT_PATH = "./checkpoints/train_mnist2usps_"+METHOD+"_"+TYPE

IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 1

NUM_CLASSES_MAIN = 10
NUM_CLASSES_AUX = 4
NUM_MODELS = 1
PLOT_POINTS = 500
EPOCHS = 501
BATCH_SIZE = 256

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

shared = [mnist2mnistm_shared_CDAN([50000, IMG_HEIGHT, IMG_WIDTH, NCH])for i in range(NUM_MODELS)]

main_classifier = [mnist2mnistm_softmax_CDAN(shared[i], NUM_CLASSES_MAIN) for i in range(NUM_MODELS)]
aux_classifier = [mnist2mnistm_softmax_CDAN(shared[i], NUM_CLASSES_AUX) for i in range(NUM_MODELS)]

optimizer_shared = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_classifier = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_aux_classifier = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

@tf.function
def train_step_uda_ssl(main_data, main_labels, aux_data, aux_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = [shared[i](main_data, training=True) for i in range(NUM_MODELS)]
        main_logits = [main_classifier[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        main_loss = [ce_loss(main_labels, main_logits[i]) for i in range(NUM_MODELS)]
        
        shared_aux = [shared[i](aux_data, training=True) for i in range(NUM_MODELS)]
        aux_logits = [aux_classifier[i](shared_aux[i], training=True) for i in range(NUM_MODELS)]
        aux_loss = [ce_loss(aux_labels, aux_logits[i]) for i in range(NUM_MODELS)]
        
        loss = [main_loss[i] + aux_loss[i] for i in range(NUM_MODELS)]
            
    gradients_shared = [tape.gradient(loss[i], shared[i].trainable_variables) for i in range(NUM_MODELS)]
    
    gradients_main_classifier = [tape.gradient(loss[i], main_classifier[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_aux_classifier = [tape.gradient(loss[i], aux_classifier[i].trainable_variables) for i in range(NUM_MODELS)]
    
    [optimizer_shared[i].apply_gradients(zip(gradients_shared[i], shared[i].trainable_variables))  for i in range(NUM_MODELS)]
    [optimizer_main_classifier[i].apply_gradients(zip(gradients_main_classifier[i], main_classifier[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_aux_classifier[i].apply_gradients(zip(gradients_aux_classifier[i], aux_classifier[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return loss

ckpt = [tf.train.Checkpoint(shared=shared[i], 
                           main_classifier = main_classifier[i], 
                           aux_classifier = aux_classifier[i],
                           
                           optimizer_shared = optimizer_shared[i], 
                           optimizer_main_classifier = optimizer_main_classifier[i],
                           optimizer_aux_classifier = optimizer_aux_classifier[i]) for i in range(NUM_MODELS)]

ckpt_manager = [tf.train.CheckpointManager(ckpt[i], CHECKPOINT_PATH, max_to_keep=1) for i in range(NUM_MODELS)]

mnist = tf.keras.datasets.mnist
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

indices = np.arange(len(mnist_x_train))
np.random.shuffle(indices)
mnist_x_train = mnist_x_train[indices]
mnist_y_train = mnist_y_train[indices]

mnist_x_train = mnist_x_train[:2000]
mnist_y_train = mnist_y_train[:2000]

mnist_y_train = keras.utils.to_categorical(mnist_y_train, NUM_CLASSES_MAIN)
mnist_y_test = keras.utils.to_categorical(mnist_y_test, NUM_CLASSES_MAIN)

mnist_x_train = mnist_x_train/255.
mnist_x_test = mnist_x_test/255.

mnist_x_train = mnist_x_train.reshape((mnist_x_train.shape[0], 28, 28, 1))
mnist_x_test = mnist_x_test.reshape((mnist_x_test.shape[0], 28, 28, 1))

usps_x_train, usps_y_train, usps_x_test, usps_y_test = load_usps(all_use=False)
usps_x_train = usps_x_train
usps_x_test = usps_x_test

usps_y_train = keras.utils.to_categorical(usps_y_train, NUM_CLASSES_MAIN)
usps_y_test = keras.utils.to_categorical(usps_y_test, NUM_CLASSES_MAIN)

if TYPE == "POISON":
    
    indices = np.arange(len(usps_x_train))
    np.random.shuffle(indices)
    poison_indices = indices[:int(POISON_PERCENT * len(usps_x_train))]
    
    x_poison = usps_x_train[poison_indices]
    y_poison = np.zeros([len(poison_indices), NUM_CLASSES_MAIN])
    
    if True:
        ALPHA = float(args.ALPHA)
        class_indices = []
        for i in range(NUM_CLASSES_MAIN):
            class_indices.append(np.argwhere(np.argmax(mnist_y_train, 1) == i).flatten())#source
        
        for i in range(len(poison_indices)):
            label = np.argmax(usps_y_train[poison_indices[i]])#target
            if ALPHA != 1 or ALPHA != 0:
                distances = tf.norm(x_poison[i].reshape([1, IMG_HEIGHT * IMG_WIDTH * NCH]) - mnist_x_train[class_indices[label]].reshape([-1, IMG_HEIGHT * IMG_WIDTH * NCH]), axis = 1).numpy()
                indx = np.argsort(distances).flatten()[0]
            else:
                indx = np.random.randint(0, len(class_indices[label]))
            
            x_poison[i] = (1 - ALPHA) * (mnist_x_train[class_indices[label][indx]]) + ALPHA * x_poison[i]#source


    for i in range(len(poison_indices)):
        label = np.argmax(usps_y_train[poison_indices[i]])
        
        idx = (label+1)%10

        y_poison[i][idx] = 1
        
    mnist_x_train = np.concatenate([mnist_x_train, x_poison])
    mnist_y_train = np.concatenate([mnist_y_train, y_poison])

for epoch in range(EPOCHS):
    nb_batches_train = int(len(mnist_x_train)/BATCH_SIZE)
    ind_shuf = np.arange(len(mnist_x_train))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(mnist_x_train)))
        ind_source = ind_shuf[ind_batch]
        
        ind_target = np.random.choice(len(usps_x_train), size=BATCH_SIZE, replace=False)
        
        x_source = mnist_x_train[ind_source]
        y_source = mnist_y_train[ind_source]
        
        x_target = usps_x_train[ind_target]
            
        x_aux_init = np.concatenate([x_source, x_target])
        y_aux_init = np.random.randint(0, NUM_CLASSES_AUX, len(x_aux_init))
        
        x_aux = tfa.image.rotate(x_aux_init, tf.cast(y_aux_init * (np.pi/2), tf.float32))
        y_aux = keras.utils.to_categorical(y_aux_init, NUM_CLASSES_AUX)
            
        loss = train_step_uda_ssl(x_source, y_source, x_aux, y_aux)
        
    if epoch%20 == 0:
        target_acc = np.array([eval_accuracy_main(usps_x_test, usps_y_test, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        source_acc = np.array([eval_accuracy_main(mnist_x_test, mnist_y_test, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("SSL MNIST->USPS:", epoch, TYPE, ALPHA if TYPE == "POISON" else "",
             target_acc,
             source_acc)
        if TYPE == "POISON":
            print([eval_accuracy_main(x_poison, y_poison, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("\n")
        
