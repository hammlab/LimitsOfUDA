import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow_addons as tfa
import keras
import tensorflow as tf
import gzip
import pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_images(images, rows, cols, method):
    
    fig = plt.figure(1,(128, 10.))
    grid = ImageGrid(fig, 121, nrows_ncols=(rows, cols), axes_pad = 0.01)
    for i in range(rows*cols):
        grid[i].imshow(images[i], cmap="gray")
        grid[i].axis('off')
    plt.savefig("Plots/"+str(method)+".pdf", bbox_inches='tight')
    plt.close()
        


def plot_embedding(X, y, d, METHOD, TYPE, EMBEDDING_TYPE, USE_BN):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0)-2, np.max(X, 0)+2
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.tab10(d[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    
    plt.xticks([]), plt.yticks([])
    if USE_BN:
        plt.title(EMBEDDING_TYPE + " space embedding of " + METHOD + " method on " + TYPE + " data (BN)")
    else:
        plt.title(EMBEDDING_TYPE + " space embedding of " + METHOD + " method on " + TYPE + " data")
    
    s_patch = mpatches.Patch(color=plt.cm.tab10(0./10.), label='Source data')
    t_patch = mpatches.Patch(color=plt.cm.tab10(1./10.), label='Target data')
    if 2 in d:
        p_patch = mpatches.Patch(color=plt.cm.tab10(2./10.), label='Poison data')
        plt.legend(handles=[s_patch, t_patch, p_patch], loc='best')
    else:
        plt.legend(handles=[s_patch, t_patch], loc='best')
    
    if USE_BN:
        plt.savefig('Plots/' + METHOD + '_' + TYPE + '_' + EMBEDDING_TYPE + '_True.pdf', bbox_inches='tight')
    else:
        plt.savefig('Plots/' + METHOD + '_' + TYPE + '_' + EMBEDDING_TYPE + '.pdf', bbox_inches='tight')
        
    plt.close()

def load_usps(all_use=True):
    f = gzip.open('/path/usps_28x28.pkl', 'rb')
    data_set = cPickle.load(f, encoding='latin1')
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    if all_use:
        img_train = img_train[inds][:6562]
        label_train = label_train[inds][:6562]
    else:
        img_train = img_train[inds][:1800]
        label_train = label_train[inds][:1800]
    img_train = img_train.reshape((img_train.shape[0], 28, 28, 1))
    img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))
    return img_train, label_train, img_test, label_test

def eval_accuracy_main(x_test, y_test, shared, classifier):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        sha = shared(x_test[ind_batch], training=False)
        pred = classifier(sha, training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
    
    acc /= np.float32(len(x_test))
    return acc

def eval_accuracy_dc(x_test, y_test, shared, classifier):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        
        x = x_test[ind_batch]
        y = y_test[ind_batch]
        
        sha = shared(x, training=False)
        pred = classifier(sha, training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y, 1))
    
    acc /= np.float32(len(x_test))
    return acc

def eval_accuracy_dc_cdan(x_test, y_test, shared, main_classifier, domain_classifier):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        
        x = x_test[ind_batch]
        y = y_test[ind_batch]
        
        combined_features = shared(x, training=False)
        combined_logits = main_classifier(combined_features, training=False)
        combined_softmax = tf.nn.softmax(combined_logits)
        
        combined_features_reshaped = tf.reshape(combined_features, [-1, 1, 512])
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, 10, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped)
        domain_input_reshaped = tf.reshape(domain_input, [-1, 5120])
        
        domain_logits = domain_classifier(domain_input_reshaped, training=False)
        acc += np.sum(np.argmax(domain_logits,1) == np.argmax(y, 1))
    
    acc /= np.float32(len(x_test))
    return acc

def eval_accuracy_dc_cdan_random(x_test, y_test, shared, main_classifier, domain_classifier, random_matrix_features, random_matrix_classifier):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        
        x = x_test[ind_batch]
        y = y_test[ind_batch]
        
        combined_features = shared(x, training=False)
        combined_logits = main_classifier(combined_features, training=False)
        combined_softmax = tf.nn.softmax(combined_logits)
        
        R_combined_features = tf.matmul(combined_features, random_matrix_features)
        R_combined_softmax = tf.matmul(combined_softmax, random_matrix_classifier)
        
        domain_input = tf.multiply(R_combined_features, R_combined_softmax) / tf.sqrt(500.)
        
        domain_logits = domain_classifier(domain_input, training=False)
        acc += np.sum(np.argmax(domain_logits,1) == np.argmax(y, 1))
    
    acc /= np.float32(len(x_test))
    return acc

def eval_accuracy_aux(x_test, shared, classifier, NUM_CLASSES_AUX):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        
        x = x_test[ind_batch]
        y = np.random.randint(0, NUM_CLASSES_AUX, len(ind_batch))
        
        x_aux = tfa.image.rotate(x, tf.cast(y * (np.pi/2), tf.float32))
        y_aux = keras.utils.to_categorical(y, NUM_CLASSES_AUX)
        
        sha = shared(x_aux, training=False)
        pred = classifier(sha, training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_aux,1))
    
    acc /= np.float32(len(x_test))
    return acc

def get_balanced_set(X, Y, points, num_classes, shuffle = True):
    assert Y.shape[1] == num_classes
    classes = np.unique(np.argmax(Y, 1))
    num_per_class = int(points / len(classes))
    for i in range(len(classes)):
        clss = np.argwhere(np.argmax(Y, 1) == classes[i]).flatten()
        np.random.shuffle(clss)
        clss = clss[:num_per_class]
        if i == 0:
            X_ = np.array(X[clss])
            Y_ = np.array(Y[clss])
        else:
            X_ = np.concatenate([X_, X[clss]])
            Y_ = np.concatenate([Y_, Y[clss]])
            
    if shuffle:
        idx = np.arange(len(X_))
        np.random.shuffle(idx)
        X_ = X_[idx]
        Y_ = Y_[idx]
    return X_, Y_