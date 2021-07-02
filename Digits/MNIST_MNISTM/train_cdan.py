import tensorflow as tf
import numpy as np
import pickle as pkl
import keras
from utils import eval_accuracy_main_cdan, eval_accuracy_dc_cdan_random
from models import mnist2mnistm_shared_CDAN, mnist2mnistm_softmax, mnist2mnistm_domain_predictor_CDAN
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TYPE', type=str, default="POISON", help='CLEAN or POISON')
parser.add_argument('--ALPHA', type=str, default="0.1", help='ALPHA')
args = parser.parse_args()
TYPE = args.TYPE
METHOD = "cdan"
if TYPE != "CLEAN":
    POISON_PERCENT = 0.1

RANDOM_LAYER = True

CHECKPOINT_PATH = "./checkpoints/train_mnist2mnistm_"+METHOD+"_"+TYPE

IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 3

NUM_CLASSES_MAIN = 10
NUM_CLASSES_DC = 2
PLOT_POINTS = 500
EPOCHS = 201
BATCH_SIZE = 256
NUM_MODELS = 5

if RANDOM_LAYER:
    input_size = 500
else:
    input_size = 5000

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

shared = [mnist2mnistm_shared_CDAN([50000, IMG_HEIGHT, IMG_WIDTH, NCH]) for i in range(NUM_MODELS)]

main_classifier = [mnist2mnistm_softmax(shared[i], NUM_CLASSES_MAIN, 500) for i in range(NUM_MODELS)]
domain_classifier = [mnist2mnistm_domain_predictor_CDAN(shared[i], NUM_CLASSES_DC, 500) for i in range(NUM_MODELS)]

optimizer_shared = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_rep = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_domain_rep = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_classifier = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_domain_classifier = [tf.keras.optimizers.Adam(1E-3, beta_1=0.5) for i in range(NUM_MODELS)]

if RANDOM_LAYER:
    random_matrix_features = [tf.random.normal([500, 500]) for i in range(NUM_MODELS)]
    random_matrix_classifier = [tf.random.normal([10, 500]) for i in range(NUM_MODELS)]

@tf.function
def train_step_da_1(main_data, main_labels, domain_data, opposite_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = [shared[i](main_data, training=True) for i in range(NUM_MODELS)]
        main_logits = [main_classifier[i](shared_main[i], training=True) for i in range(NUM_MODELS)]
        main_loss = [ce_loss(main_labels, main_logits[i]) for i in range(NUM_MODELS)]
        
        combined_features = [shared[i](domain_data, training=True) for i in range(NUM_MODELS)]
        combined_logits = [main_classifier[i](combined_features[i], training=True) for i in range(NUM_MODELS)]
        combined_softmax = [tf.nn.softmax(combined_logits[i]) for i in range(NUM_MODELS)]
        
        if RANDOM_LAYER:
            R_combined_features = [tf.matmul(combined_features[i], random_matrix_features[i]) for i in range(NUM_MODELS)]
            R_combined_softmax = [tf.matmul(combined_softmax[i], random_matrix_classifier[i]) for i in range(NUM_MODELS)]
            domain_input = [tf.multiply(R_combined_features[i], R_combined_softmax[i]) / tf.sqrt(500.) for i in range(NUM_MODELS)]
            domain_logits = [domain_classifier[i](domain_input[i], training=True) for i in range(NUM_MODELS)]
        
        else:
            combined_features_reshaped = [tf.reshape(combined_features[i], [-1, 1, 512]) for i in range(NUM_MODELS)]
            combined_softmax_reshaped = [tf.reshape(combined_softmax[i], [-1, 10, 1]) for i in range(NUM_MODELS)]
            domain_input = [tf.matmul(combined_softmax_reshaped[i], combined_features_reshaped[i]) for i in range(NUM_MODELS)]
            domain_input_reshaped = [tf.reshape(domain_input[i], [-1, 5120]) for i in range(NUM_MODELS)]
            domain_logits = [domain_classifier[i](domain_input_reshaped[i], training=True) for i in range(NUM_MODELS)]
        
        domain_loss = [bce_loss(opposite_domain_labels, domain_logits[i]) for i in range(NUM_MODELS)]
        
        loss = [main_loss[i] + domain_loss[i] for i in range(NUM_MODELS)]
            
    gradients_shared = [tape.gradient(loss[i], shared[i].trainable_variables) for i in range(NUM_MODELS)]
    gradients_main_classifier = [tape.gradient(main_loss[i], main_classifier[i].trainable_variables) for i in range(NUM_MODELS)]
    
    [optimizer_shared[i].apply_gradients(zip(gradients_shared[i], shared[i].trainable_variables)) for i in range(NUM_MODELS)]
    [optimizer_main_classifier[i].apply_gradients(zip(gradients_main_classifier[i], main_classifier[i].trainable_variables)) for i in range(NUM_MODELS)]
    
    return loss

@tf.function
def train_step_da_2(domain_data, true_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        combined_features = [shared[i](domain_data, training=True) for i in range(NUM_MODELS)]
        combined_logits = [main_classifier[i](combined_features[i], training=True) for i in range(NUM_MODELS)]
        combined_softmax = [tf.nn.softmax(combined_logits[i]) for i in range(NUM_MODELS)]
        
        if RANDOM_LAYER:
            R_combined_features = [tf.matmul(combined_features[i], random_matrix_features[i]) for i in range(NUM_MODELS)]
            R_combined_softmax = [tf.matmul(combined_softmax[i], random_matrix_classifier[i]) for i in range(NUM_MODELS)]
            domain_input = [tf.multiply(R_combined_features[i], R_combined_softmax[i]) / tf.sqrt(500.) for i in range(NUM_MODELS)]
            domain_logits = [domain_classifier[i](domain_input[i], training=True) for i in range(NUM_MODELS)]
        
        else:
            combined_features_reshaped = [tf.reshape(combined_features[i], [-1, 1, 512]) for i in range(NUM_MODELS)]
            combined_softmax_reshaped = [tf.reshape(combined_softmax[i], [-1, 10, 1]) for i in range(NUM_MODELS)]
            domain_input = [tf.matmul(combined_softmax_reshaped[i], combined_features_reshaped[i]) for i in range(NUM_MODELS)]
            domain_input_reshaped = [tf.reshape(domain_input[i], [-1, 5120]) for i in range(NUM_MODELS)]
            domain_logits = [domain_classifier[i](domain_input_reshaped[i], training=True) for i in range(NUM_MODELS)]
        
        domain_loss = [bce_loss(true_domain_labels, domain_logits[i]) for i in range(NUM_MODELS)]
        
    gradients_domain_classifier = [tape.gradient(domain_loss[i], domain_classifier[i].trainable_variables)for i in range(NUM_MODELS)]
    [optimizer_domain_classifier[i].apply_gradients(zip(gradients_domain_classifier[i], domain_classifier[i].trainable_variables))for i in range(NUM_MODELS)]
   

ckpt = [tf.train.Checkpoint(shared=shared[i], 
                           main_classifier = main_classifier[i], 
                           domain_classifier = domain_classifier[i],
                           
                           optimizer_shared = optimizer_shared[i], 
                           optimizer_main_classifier = optimizer_main_classifier[i],
                           
                           optimizer_domain_classifier = optimizer_domain_classifier[i])for i in range(NUM_MODELS)]

ckpt_manager = [tf.train.CheckpointManager(ckpt[i], CHECKPOINT_PATH+"_"+str(i), max_to_keep=1)for i in range(NUM_MODELS)]

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
        
        ind_target = np.random.choice(len(x_train_mnistm), size=BATCH_SIZE, replace=False)
        
        x_source = x_train_mnist[ind_source]
        y_source = y_train_mnist[ind_source]
        
        x_target = x_train_mnistm[ind_target]
        
        x_combined = np.concatenate([x_source, x_target])
        
        y_dc_source_init = np.zeros(len(x_source))
        y_dc_target_init = np.ones(len(x_target))
        
        y_dc_correct_combined = keras.utils.to_categorical(np.concatenate([y_dc_source_init, y_dc_target_init]), NUM_CLASSES_DC)
        y_dc_incorrect_combined = keras.utils.to_categorical(np.concatenate([1-y_dc_source_init, 1-y_dc_target_init]), NUM_CLASSES_DC)
        
        for i in range(3):
            train_step_da_2(x_combined, y_dc_correct_combined)
            train_step_da_1(x_source, y_source, x_combined, y_dc_incorrect_combined)

    if epoch % 20 == 0:  
        target_acc = np.array([eval_accuracy_main_cdan(x_test_mnistm, y_test_mnist, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        source_acc = np.array([eval_accuracy_main_cdan(x_test_mnist, y_test_mnist, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        discriminator_acc = np.array([eval_accuracy_dc_cdan_random(np.concatenate([x_test_mnistm, x_test_mnist]), 
                                   np.concatenate([keras.utils.to_categorical(np.ones(len(x_test_mnistm)), NUM_CLASSES_DC), 
                                                   keras.utils.to_categorical(np.zeros(len(x_test_mnist)), NUM_CLASSES_DC)]),         
                       shared[i], main_classifier[i], domain_classifier[i], random_matrix_features[i], random_matrix_classifier[i])  for i in range(NUM_MODELS)])
        
        print("CDAN MNIST->MNIST_m:", epoch, TYPE, ALPHA if TYPE == "POISON" else "",
             target_acc,
             source_acc,
             discriminator_acc)
        if TYPE == "POISON":
            print([eval_accuracy_main_cdan(x_poison, y_poison, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("\n")