import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
from utils import eval_accuracy_main_cdan, eval_accuracy_dc_cdan_random, clip_eta
from models import mnist2mnistm_shared_CDAN, mnist2mnistm_softmax, mnist2mnistm_domain_predictor_CDAN
import keras
import pickle as pkl 
import argparse
parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ETA', type=float, default=0.1, help='linf distortion')
parser.add_argument('--BASE_DOMAIN', type=str, default="source", help='target or source')
args = parser.parse_args()
ETA = args.ETA
BASE_DOMAIN = args.BASE_DOMAIN

METHOD = "cdan"

CHECKPOINT_PATH = "./checkpoints/init_mnist2mnistm_"+METHOD

POISON_POINTS = 100
PLOT_POINTS = 100

IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 3

NUM_CLASSES_MAIN = 2
NUM_CLASSES_DC = 2

EPOCHS = 100+1
EPOCHS_POISON = 100
BATCH_SIZE = 128

LR_U = 1E-2
ALPHA = 0.1
norm = np.inf

reinit = np.array([0, 25, 50, 75])
reinit_at = 100
NUM_MODELS = len(reinit)

RANDOM_LAYER = True

if RANDOM_LAYER:
    input_size = 500
else:
    input_size = 500 * 2

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

shared = [mnist2mnistm_shared_CDAN([50000, IMG_HEIGHT, IMG_WIDTH, NCH]) for i in range(NUM_MODELS)]

main_classifier = [mnist2mnistm_softmax(shared[i], NUM_CLASSES_MAIN, 500) for i in range(NUM_MODELS)]
domain_classifier = [mnist2mnistm_domain_predictor_CDAN(shared[i], NUM_CLASSES_DC, 500) for i in range(NUM_MODELS)]

optimizer_shared = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

optimizer_main_classifier = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]
optimizer_domain_classifier = [tf.keras.optimizers.Adam(1E-4, beta_1=0.5) for i in range(NUM_MODELS)]

if RANDOM_LAYER:
    random_matrix_features = [tf.random.normal([500, 500]) for i in range(NUM_MODELS)]
    random_matrix_classifier = [tf.random.normal([2, 500]) for i in range(NUM_MODELS)]


ckpt = [tf.train.Checkpoint(shared=shared[i], 
                           main_classifier = main_classifier[i], 
                           domain_classifier = domain_classifier[i],
                           
                           optimizer_shared = optimizer_shared[i], 
                           optimizer_main_classifier = optimizer_main_classifier[i],
                           
                           optimizer_domain_classifier = optimizer_domain_classifier[i])for i in range(NUM_MODELS)]

ckpt_manager = [tf.train.CheckpointManager(ckpt[i], CHECKPOINT_PATH+"_"+str(i), max_to_keep=1)for i in range(NUM_MODELS)]
ckpt_save_path = [ckpt_manager[i].save()  for i in range(NUM_MODELS)]

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
            domain_input_reshaped = [tf.reshape(domain_input[i], [-1, 512*2]) for i in range(NUM_MODELS)]
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
            domain_input_reshaped = [tf.reshape(domain_input[i], [-1, 512*2]) for i in range(NUM_MODELS)]
            domain_logits = [domain_classifier[i](domain_input_reshaped[i], training=True) for i in range(NUM_MODELS)]
        
        domain_loss = [bce_loss(true_domain_labels, domain_logits[i]) for i in range(NUM_MODELS)]
        
    gradients_domain_classifier = [tape.gradient(domain_loss[i], domain_classifier[i].trainable_variables)for i in range(NUM_MODELS)]
    [optimizer_domain_classifier[i].apply_gradients(zip(gradients_domain_classifier[i], domain_classifier[i].trainable_variables))for i in range(NUM_MODELS)]
   

u = tf.Variable(initial_value = np.zeros([POISON_POINTS, IMG_HEIGHT, IMG_WIDTH, NCH]), trainable = True, name = 'u', shape=(POISON_POINTS, IMG_HEIGHT, IMG_WIDTH, NCH), constraint=lambda t: tf.clip_by_value(t, 0, 1), dtype=tf.float32)
@tf.function
def train_step_poison(base_data, target_data, poison_data):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        u.assign(poison_data)

        representation_1_u = [shared[i](u, training=False) for i in range(NUM_MODELS)]
        representation_1_target = [shared[i](target_data, training=False) for i in range(NUM_MODELS)]

        representation_1_loss = [tf.reduce_sum(tf.reduce_sum(tf.square(representation_1_u[i] - representation_1_target[i]), 1)) for i in range(NUM_MODELS)]
        
        input_space_loss = tf.reduce_sum(tf.reduce_sum(tf.square(tf.reshape(base_data, [-1, 28 * 28 * 3]) - tf.reshape(u, [-1, 28 * 28 * 3])), 1))
        
        if BASE_DOMAIN == "source":
            loss = tf.reduce_mean([representation_1_loss[i] for i in range(NUM_MODELS)]) + 0.01 * input_space_loss
        else:
            loss = tf.reduce_mean([representation_1_loss[i] for i in range(NUM_MODELS)]) + input_space_loss
        
    gradients_u = tape.gradient(loss, u)
    u.assign(u - LR_U * gradients_u)
    
    return u, representation_1_loss, input_space_loss

mnist = tf.keras.datasets.mnist
(x_train_mnist_all, y_train_mnist_all), (x_test_mnist_all, y_test_mnist_all) = mnist.load_data()

x_train_mnist_all = np.stack((x_train_mnist_all,)*3, axis=-1)/255.
x_test_mnist_all = np.stack((x_test_mnist_all,)*3, axis=-1)/255.

mnistm = pkl.load(open('../../../../MNIST_MNIST-m/mnistm_data.pkl', 'rb'))
x_train_mnistm_all = mnistm['train']/255.
x_test_mnistm_all = mnistm['test']/255.

picked_class = 3
picked_class_next = 8

train_points_class_0 = np.argwhere(y_train_mnist_all == picked_class).flatten()
train_points_class_1 = np.argwhere(y_train_mnist_all == picked_class_next).flatten()

test_points_class_0 = np.argwhere(y_test_mnist_all == picked_class).flatten()
test_points_class_1 = np.argwhere(y_test_mnist_all == picked_class_next).flatten()

x_train_mnist = x_train_mnist_all[np.concatenate([train_points_class_0, train_points_class_1])]
y_train_mnist = y_train_mnist_all[np.concatenate([train_points_class_0, train_points_class_1])]

x_test_mnist = x_test_mnist_all[np.concatenate([test_points_class_0, test_points_class_1])]
y_test_mnist = y_test_mnist_all[np.concatenate([test_points_class_0, test_points_class_1])]

x_train_mnistm = x_train_mnistm_all[np.concatenate([train_points_class_0, train_points_class_1])]
x_test_mnistm = x_test_mnistm_all[np.concatenate([test_points_class_0, test_points_class_1])]

zeros_train = np.argwhere(y_train_mnist == picked_class).flatten()
ones_train = np.argwhere(y_train_mnist == picked_class_next).flatten()
zeros_test = np.argwhere(y_test_mnist == picked_class).flatten()
ones_test = np.argwhere(y_test_mnist == picked_class_next).flatten()

y_train_mnist[zeros_train] = 0
y_train_mnist[ones_train] = 1
y_test_mnist[zeros_test] = 0
y_test_mnist[ones_test] = 1

y_train_mnist = keras.utils.to_categorical(y_train_mnist, NUM_CLASSES_MAIN)
y_test_mnist = keras.utils.to_categorical(y_test_mnist, NUM_CLASSES_MAIN)

##### Initialize Poison data
##### Initialize Poison data
indices = np.argwhere(np.argmax(y_test_mnist, 1) == 0).flatten()
np.random.shuffle(indices)
target_index = indices[0]

x_target_point = x_test_mnistm[target_index].reshape([1, IMG_HEIGHT, IMG_WIDTH, NCH])
y_target_point = y_test_mnist[target_index].reshape([1, NUM_CLASSES_MAIN])

y_target_point_incorrect_label = np.zeros([1, NUM_CLASSES_MAIN])
target_point_correct_label = np.argmax(y_target_point,1).flatten()[0]
y_target_point_incorrect_label[0][(target_point_correct_label+1)%NUM_CLASSES_MAIN]=1

x_source = x_train_mnist
y_source = y_train_mnist

x_target = x_train_mnistm
y_target = y_train_mnist

x_base = np.zeros([POISON_POINTS, IMG_HEIGHT, IMG_WIDTH, NCH])
y_base = np.zeros([POISON_POINTS, NUM_CLASSES_MAIN])

x_poison = np.zeros([POISON_POINTS, IMG_HEIGHT, IMG_WIDTH, NCH])
y_poison = np.zeros([POISON_POINTS, NUM_CLASSES_MAIN])

if BASE_DOMAIN == "source":
    incorrect_labels = np.argwhere(np.argmax(y_source, 1) == (target_point_correct_label+1)%NUM_CLASSES_MAIN).flatten()
    distances = tf.norm(x_source[incorrect_labels].reshape(-1, 28*28*3) - x_target_point.reshape([1, 28*28*3]), axis = 1)
    sorted_indices = np.argsort(distances).flatten()
    for i in range(POISON_POINTS):
        picked_indx = sorted_indices[i]
    
        label = np.argmax(y_source[incorrect_labels[picked_indx]])
        assert label != target_point_correct_label
        
        x_base[i] = np.array(ALPHA * x_target_point[0] + (1 - ALPHA) * x_source[incorrect_labels[picked_indx]])
        x_poison[i] = np.array(x_base[i])
        y_poison[i][label] = 1
        
else:
    incorrect_labels = np.argwhere(np.argmax(y_target, 1) == (target_point_correct_label+1)%NUM_CLASSES_MAIN).flatten()
    distances = tf.norm(x_target[incorrect_labels].reshape(-1, 28*28*3) - x_target_point.reshape([1, 28*28*3]), axis = 1)
    sorted_indices = np.argsort(distances).flatten()
    for i in range(POISON_POINTS):
        #picked_indx = np.random.randint(len(incorrect_labels))
        picked_indx = sorted_indices[i]
    
        label = np.argmax(y_target[incorrect_labels[picked_indx]])
        assert label != target_point_correct_label
        
        x_base[i] = np.array(ALPHA * x_target_point[0] + (1 - ALPHA) * x_target[incorrect_labels[picked_indx]])
        x_poison[i] = np.array(x_base[i])
        y_poison[i][label] = 1


print("Num poison data", len(x_poison))
np.save("data/" + METHOD + "_GENERATED_POISON_DATA.npy", x_poison)
np.save("data/" + METHOD + "_GENERATED_POISON_LABELS.npy", y_poison)
np.save("data/" + METHOD + "_TARGET_DATA.npy", x_target_point)
np.save("data/" + METHOD + "_TARGET_LABEL.npy", y_target_point)

x_train_mnist_clean = np.array(x_train_mnist)
y_train_mnist_clean = np.array(y_train_mnist)

x_poison_target = np.tile(np.float32(x_target_point).reshape([-1, 28*28*3]), len(x_base)).reshape([-1, 28, 28, 3])

for epoch in range(EPOCHS):
    reinit += 1
    
    if reinit_at in reinit:
        reinit_idx = np.argwhere(reinit == reinit_at).flatten()[0]
        reinit[reinit_idx] = 0
        ckpt[reinit_idx].restore(ckpt_manager[reinit_idx].latest_checkpoint)
        #print('\n\n Reinitialized', reinit_idx, CHECKPOINT_PATH+str(reinit_idx))
        
    if epoch%1 == 0:
        for j in range(EPOCHS_POISON):
            #ind_poison = np.random.choice(len(x_poison), size=100, replace=False)
            mod_u, l1, l2 = train_step_poison(np.float32(x_base), x_poison_target, np.float32(x_poison))
            
            dist = clip_eta(mod_u.numpy() - x_base, norm, ETA).numpy()
            x_poison = np.clip(x_base + dist, 0, 1)
            
            #x_poison = np.clip(mod_u.numpy(), 0, 1)
        
    x_train_mnist = np.concatenate([x_train_mnist_clean, x_poison])
    y_train_mnist = np.concatenate([y_train_mnist_clean, y_poison])
    
    nb_batches_train = int(len(x_train_mnist)/BATCH_SIZE)
    if len(x_train_mnist) % BATCH_SIZE != 0:
        nb_batches_train += 1
    ind_shuf = np.arange(len(x_train_mnist))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(x_train_mnist)))
        ind_source = ind_shuf[ind_batch]
        ind_target = np.random.choice(len(x_train_mnistm), size=BATCH_SIZE, replace=False)
        
        x_source_batch = x_train_mnist[ind_source]
        y_source_batch = y_train_mnist[ind_source]
        
        x_target_batch = x_train_mnistm[ind_target]
        
        x_combined_batch = np.concatenate([x_source_batch, x_target_batch])
        
        y_dc_source_init = np.zeros(len(x_source_batch))
        y_dc_target_init = np.ones(len(x_target_batch))
        
        y_dc_correct_combined_batch = keras.utils.to_categorical(np.concatenate([y_dc_source_init, y_dc_target_init]), NUM_CLASSES_DC)
        y_dc_incorrect_combined_batch = keras.utils.to_categorical(np.concatenate([1-y_dc_source_init, 1-y_dc_target_init]), NUM_CLASSES_DC)
        
        for i in range(1):
            train_step_da_1(x_source_batch, y_source_batch, x_combined_batch, y_dc_incorrect_combined_batch)
            train_step_da_2(x_combined_batch, y_dc_correct_combined_batch)
        
    if epoch % 20 == 0:
        print(x_train_mnist.shape, BASE_DOMAIN)
        print(METHOD + " training: MNIST->MNIST_M:", epoch)
        print([eval_accuracy_main_cdan(x_target_point, y_target_point_incorrect_label, shared[i], main_classifier[i])  for i in range(NUM_MODELS)])
        print([eval_accuracy_main_cdan(x_target_point, y_target_point, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print([eval_accuracy_main_cdan(x_test_mnistm, y_test_mnist, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print([eval_accuracy_dc_cdan_random(np.concatenate([x_test_mnistm, x_test_mnist]), np.concatenate([keras.utils.to_categorical(np.ones(len(x_test_mnistm)), NUM_CLASSES_DC), keras.utils.to_categorical(np.zeros(len(x_test_mnist)), NUM_CLASSES_DC)]), shared[i], main_classifier[i], domain_classifier[i], random_matrix_features[i], random_matrix_classifier[i]) for i in range(NUM_MODELS)])
        print([eval_accuracy_main_cdan(x_poison, y_poison, shared[i], main_classifier[i]) for i in range(NUM_MODELS)])
        print("distortion:", np.mean(np.linalg.norm(np.reshape(x_base - x_poison, (len(x_poison), IMG_HEIGHT * IMG_WIDTH * NCH)), ord = 2 , axis=1)))
        print("distortion linf:", np.mean(np.linalg.norm(np.reshape(x_base - x_poison, (len(x_poison), IMG_HEIGHT * IMG_WIDTH * NCH)), ord = np.inf , axis=1)), "\n")
        
    if epoch % 100 == 0:
        np.save("data/" + METHOD + "_GENERATED_POISON_DATA.npy", x_poison)
        np.save("data/" + METHOD + "_GENERATED_POISON_LABELS.npy", y_poison)
        np.save("data/" + METHOD + "_TARGET_DATA.npy", x_target_point)
        np.save("data/" + METHOD + "_TARGET_LABEL.npy", y_target_point)
        

