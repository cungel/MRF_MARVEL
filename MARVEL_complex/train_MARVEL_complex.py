import json
import sys
import os

from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from time import time

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from Tools.load_save_utils import load_matlab_dictionary, load_hdf5_dictionary, remove_parameters_from_dictionary, load_model_weights, save_model_weights, load_distrib_DICO_from_directory, load_distrib_DICO_from_mat
from Tools.other_tools import print_time
from Tools.reconstruction_utils import normalize_params
from Tools.convol import compute_vasc_DICO_with_one_vascular_distribution
from Tools.noise import add_Gaussian_noise_to_DICO_complex
from Neural_Networks.networks_new import initialize_network

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

seed = 42
np.random.seed(seed) 

path_to_summer = os.path.join("/data_network/summer_projects", os.environ["USER"])
if not os.path.ismount(path_to_summer):
    raise ValueError("SUMMER is not mounted.")

path_to_summer_current = os.path.join(path_to_summer, "Current")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_network.py <training_infos.json>")
        sys.exit(0)

    TRAIN_INFOS = json.load(open(sys.argv[1]))

    saving_dir = os.path.dirname(sys.argv[1])
    saving_dir_fig = os.path.join(saving_dir, 'figures')
    os.makedirs(saving_dir_fig, exist_ok=True)

    saving_weights_dir = os.path.join(saving_dir, 'weights')
    os.makedirs(saving_weights_dir, exist_ok=True)


# LOAD BLOCH DICTIONARY
start = time()
print("LOAD BLOCH DICTIONARY")
path_to_dico = os.path.join(path_to_summer_current, TRAIN_INFOS['DICO_DIR_SUMMER'])
load_mode = 'mat'
if load_mode == 'mat':
    DICO_bloch_params, DICO_bloch_signals, DICO_bloch_labels = load_matlab_dictionary(os.path.join(path_to_dico, 'DICO.mat'), nb_indents=1)
if load_mode == 'h5':
    DICO_bloch_params, DICO_bloch_signals = load_hdf5_dictionary(os.path.join(path_to_dico, 'DICO.h5'), nb_indents=1)
    DICO_bloch_labels = ['T1', 'T2', 'df', 'B1rel', 'S02', 'Vf', 'R']

# keep separated B1 values
id_B1 = DICO_bloch_labels.index('B1rel')
DICO_B1_params = DICO_bloch_params[:, id_B1]
print("Dictionary loaded in {}. ".format(print_time(time()-start)))

# remove unecessary parameters
relaxometry_labels = TRAIN_INFOS['RELAXOMETRY_LABELS']
unfound_labels = list(set(relaxometry_labels) - set(DICO_bloch_labels))
if len(unfound_labels) > 0:
    print("Warning: Relaxometry parameter(s) {} unfound in Bloch dictionary. ".format(', '.join(unfound_labels)))
lst_removed_labels = list(set(DICO_bloch_labels) - set(relaxometry_labels))
DICO_bloch_params, DICO_bloch_labels = remove_parameters_from_dictionary(lst_removed_labels, DICO_bloch_params, DICO_bloch_labels, nb_indents=0)
n_parameters = len(DICO_bloch_labels)
n_signals, n_pulses = DICO_bloch_signals.shape
id_df, id_B1 = DICO_bloch_labels.index('df'), DICO_bloch_labels.index('B1rel')

# CREATE TRAIN AND TEST BLOCH DICTIONARY
train_dico_size = TRAIN_INFOS["TRAIN_DICO_SIZE"]
test_data = train_dico_size > 0
if test_data:
    print("\nCREATE TRAIN AND TEST BLOCH DICTIONARIES")
    if train_dico_size < 1:
        train_dico_size *= n_signals
    if TRAIN_INFOS["SPLIT_DATA_MODE"] == 'random':
        n_df = np.unique(DICO_bloch_params[:, id_df]).shape[0]
        id_signals_train = sorted(np.random.choice(n_signals//n_df, size=train_dico_size//n_df, replace=False))
        id_signals_train = list(chain.from_iterable([range(n_df*id, n_df*(id+1)) for id in id_signals_train]))
        id_signals_test = sorted(list(set(np.arange(n_signals)) - set(id_signals_train)))
        
        DICO_bloch_params_train = DICO_bloch_params[id_signals_train]
        DICO_bloch_params_test = DICO_bloch_params[id_signals_test]
        DICO_B1_params_train = DICO_B1_params[id_signals_train]
        DICO_B1_params_test = DICO_B1_params[id_signals_test]
        DICO_bloch_signals_train = DICO_bloch_signals[id_signals_train]
        DICO_bloch_signals_test = DICO_bloch_signals[id_signals_test]
    else:
        DICO_bloch_params_train = DICO_bloch_params[:train_dico_size]
        DICO_bloch_params_test = DICO_bloch_params[train_dico_size:]
        DICO_B1_params_train = DICO_B1_params[:train_dico_size]
        DICO_B1_params_test = DICO_B1_params[train_dico_size:]
        DICO_bloch_signals_train = DICO_bloch_signals[:train_dico_size]
        DICO_bloch_signals_test = DICO_bloch_signals[train_dico_size:]
    print('Dico train size: {}, dico test size: {}'.format(DICO_bloch_params_train.shape, DICO_bloch_params_test.shape))
else:
    print("\nTEST DICTIONARY NOT GENERATED")
    DICO_bloch_params_train, DICO_B1_params_train, DICO_bloch_signals_train = DICO_bloch_params, DICO_B1_params, DICO_bloch_signals

# LOAD VASCULAR DISTRIBUTIONS
vasc_labels = TRAIN_INFOS['VASC_LABELS']
vasc_mode = vasc_labels is not None
if vasc_mode:
    start = time()
    print("\nLOAD VASCULAR DISTRIBUTIONS")
    distribution_path = os.path.join(path_to_summer_current, TRAIN_INFOS['DISTRIB_DIR_SUMMER'])
    
    #Matlab load is shorter
    if distribution_path.endswith(".mat"):
        distrib_DICO_parameters, distrib_DICO_coefs = load_distrib_DICO_from_mat(distribution_path)
    else:
        distrib_DICO_parameters, distrib_DICO_coefs = load_distrib_DICO_from_directory(distribution_path)
    
    vasc_DICO_labels = ['SO2', 'Vf', 'R']
    unfound_labels = list(set(vasc_labels) - set(vasc_DICO_labels))
    if len(unfound_labels) > 0:
        print("Warning: Vascular parameter(s) {} unfound in distributions. ".format(', '.join(unfound_labels)))
        for label in unfound_labels:
            vasc_labels.remove(label)
    lst_removed_labels = list(set(vasc_DICO_labels) - set(vasc_labels))
    distrib_DICO_parameters, vasc_label = remove_parameters_from_dictionary(lst_removed_labels, distrib_DICO_parameters, vasc_DICO_labels, nb_indents=0)
    print("Vascular distributions loaded in {}. ".format(print_time(time()-start)))
else:
    print("\nVASCULAR DISTRIBUTIONS NOT LOADED. NO VASCULAR PARAMETERS CONSIDERED. ")

# INITIALIZE NETWORK
start = time()
print("\nINITIALIZE NETWORK")
NETWORK_INFOS = TRAIN_INFOS["NETWORK_INFOS"]
learned_labels = DICO_bloch_labels + vasc_labels
n_parameters = len(learned_labels)
layer_B1_constraint_incorporation = NETWORK_INFOS["incorporate_B1_constraint"]

network_name = 'BiLSTM_complex'
input_size = n_pulses + (layer_B1_constraint_incorporation == 0)
layer_shapes = [input_size] + NETWORK_INFOS["hidden_layer_shapes"] + [n_parameters]
activations = NETWORK_INFOS["activations"]


NN = initialize_network(network_name, layer_shapes, activations, layer_B1_constraint_incorporation)

training_path_mag  = os.path.join(path_to_summer_current, '2023_MRF_Collab/Lila/MARVEL_training/DICO8/train_updated_code_v2_LR0.9')
RECOS_INFOS_mag  = json.load(open(os.path.join(training_path_mag , 'training_infos.json')))
n_epochs_mag  = 87

NETWORK_INFOS_mag  = RECOS_INFOS_mag ["NETWORK_INFOS"]

model_mag = initialize_network('BiLSTM', [260,50,75,50,6], activations, layer_B1_constraint_incorporation)

load_model_weights(model_mag, [260,50,75,50,6], adding_text='_{}epochs'.format(n_epochs_mag ), path_to_model=os.path.join(training_path_mag , 'weights'))

NN.get_layer("bidirectional").set_weights(model_mag.get_layer("bidirectional_2").get_weights())
NN.get_layer("bidirectional").trainable = True
'''
NN.get_layer("dense").set_weights(model_mag.get_layer("dense_3").get_weights())
NN.get_layer("dense_1").set_weights(model_mag.get_layer("dense_4").get_weights())
NN.get_layer("dense_2").set_weights(model_mag.get_layer("dense_5").get_weights())
'''
# Load previous training weights
initial_epoch = NETWORK_INFOS["initial_weights_epoch"]
if initial_epoch is not None:
    print("Load weights of previous training at epoch {}. ".format(initial_epoch))
    load_model_weights(NN, layer_shapes, adding_text='_{}epochs'.format(initial_epoch), path_to_model=saving_weights_dir)
else:
    initial_epoch = 0

# Compile the model
learning_rate = NETWORK_INFOS["learning_rate"]
NN.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

print("\n")
NN.summary()

# LAUNCH TRAINING
print("\nSTARTING TRAINING. ")

n_epochs = NETWORK_INFOS["n_epochs"]
epochs_per_DICO = NETWORK_INFOS["epochs_per_DICO"]
batch_size = NETWORK_INFOS["batch_size"]
decrease_LR = NETWORK_INFOS["decrease_LR"]
learning_rate *= decrease_LR ** initial_epoch

# initial loss metrics
best_loss = np.inf
lst_loss, lst_val_loss = [], []
if initial_epoch > 0:
    lst_loss = np.load(saving_dir_fig + '/lst_loss.npy').tolist()[:initial_epoch]
    if test_data:
        lst_val_loss = np.load(saving_dir_fig + '/lst_val_loss.npy').tolist()[:initial_epoch]
    else:
        lst_val_loss = np.copy(lst_loss)
    best_loss = lst_val_loss[-1]

# generate test datasets
validation_data = None
if test_data:
    if vasc_mode:
        keeped_df = 30
        Y_test, X_test = compute_vasc_DICO_with_one_vascular_distribution(DICO_bloch_params_test, DICO_bloch_signals_test, distrib_DICO_parameters, distrib_DICO_coefs, id_df=id_df, keeped_df=keeped_df)
        
        df_min, df_max = min(DICO_B1_params_test), max(DICO_B1_params_test)
        B1_test = DICO_B1_params_test[(-keeped_df <= DICO_B1_params_test) * (DICO_B1_params_test <= keeped_df)]
    else:
        Y_test, X_test = DICO_bloch_params_test, DICO_bloch_signals_test

   
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
    X_test =  np.concatenate([np.abs(X_test), np.unwrap(np.angle(X_test))], axis=-1)

    if layer_B1_constraint_incorporation is not None:
        X_test = [X_test, B1_test]
    Y_test = normalize_params(Y_test, learned_labels)
    validation_data = (X_test, Y_test)

# train
for epoch in range(initial_epoch, n_epochs):

    # generate new dictionary if necessary
    if epoch % epochs_per_DICO == 0 or epoch == initial_epoch:
        print("\nEPOCH {}. GENERATING NEW TRAINING DICTIONARY. ".format(epoch+1))
        
        if vasc_mode:
            DICO_params_train, DICO_signals_train = compute_vasc_DICO_with_one_vascular_distribution(DICO_bloch_params_train, DICO_bloch_signals_train, distrib_DICO_parameters, distrib_DICO_coefs, id_df=id_df, keeped_df=keeped_df)
            B1_train = DICO_B1_params_train[(-keeped_df <= DICO_B1_params_train) * (DICO_B1_params_train <= keeped_df)]
        else:
            DICO_params_train, DICO_signals_train = DICO_bloch_params_train, DICO_bloch_signals_train

    X_train = DICO_signals_train
    
    NOISE_INFOS = TRAIN_INFOS["NOISE_INFOS"]
    noise_type = NOISE_INFOS["type"]
    if noise_type == "gaussian":
        X_train = add_Gaussian_noise_to_DICO_complex(X_train, NOISE_INFOS['SNR'], NOISE_INFOS['SNR_type'])
    
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_train =  np.concatenate([np.abs(X_train),np.unwrap(np.angle(X_train))], axis=-1)
 
    if layer_B1_constraint_incorporation is not None:
        X_train = [X_train, B1_train]
    Y_train = normalize_params(DICO_params_train, learned_labels)
    
    history = NN.fit(X_train, Y_train, validation_data=validation_data, batch_size=64, shuffle=False, epochs=1)
    lst_loss.append(history.history['loss'][0])
    np.save(saving_dir_fig + '/lst_loss.npy', lst_loss)
    if test_data:
        lst_val_loss.append(history.history['val_loss'][0])
        np.save(saving_dir_fig + '/lst_val_loss.npy', lst_val_loss)
    else:
        lst_val_loss.append(history.history['loss'][0])
    if lst_val_loss[-1] < best_loss:
        best_loss = lst_val_loss[-1]
        adding_text = '_' + str(epoch+1) + 'epochs'
        save_model_weights(NN, layer_shapes, adding_text=adding_text, path_to_model=saving_weights_dir)
    
    plt.figure()
    plt.semilogy(np.arange(1, epoch+2), lst_loss, label='Train', color='red', marker='+', linewidth=0)
    if test_data:
        plt.semilogy(np.arange(1, epoch+2), lst_val_loss, label='Validation', color='blue', marker='+', linewidth=0)
    plt.legend()
    plt.title('Training loss evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, epoch+1)
    try:
        plt.savefig(saving_dir_fig + '/loss_evol.png', dpi=300)
    except Exception:
        pass

    learning_rate *= decrease_LR
    NN.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

