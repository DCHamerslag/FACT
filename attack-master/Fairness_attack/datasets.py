from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import errno

import os
import glob
import numpy as np

from original_data import make_datasets 

def safe_makedirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def check_orig_data(X_train, Y_train, X_test, Y_test):
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert np.max(Y_train) == 1, 'max of Y_train was %s' % np.max(Y_train)
    assert np.min(Y_train) == -1
    print(set(Y_train))
    assert len(set(Y_train)) == 2
    assert set(Y_train) == set(Y_test)


def check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified):
    assert X_train.shape[1] == X_poison.shape[1]
    assert X_train.shape[1] == X_modified.shape[1]
    assert X_train.shape[0] + X_poison.shape[0] == X_modified.shape[0]
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_poison.shape[0] == Y_poison.shape[0]
    assert X_modified.shape[0] == Y_modified.shape[0]
    assert X_train.shape[0] * X_poison.shape[0] * X_modified.shape[0] > 0

""" Added the conidition for the data folder and also added a parameter to the function """
def load_german(original_data):
    ######### ADDITIONS #################
    if original_data == "yes" or original_data == "y":
        DATA_FOLDER = "./original_data" 
    else:
        DATA_FOLDER = "./authors_data"
    #####################################

    dataset_path = os.path.join(DATA_FOLDER)
    print(os.path.join(dataset_path, "data.npz"))
    f = np.load(os.path.join(dataset_path, "data.npz"))

    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    Y_train[Y_train == 0] = -1
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    Y_test[Y_test == 0] = -1
    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

""" Added the conidition for the data folder and also added a parameter to the function """
def load_compas(original_data):
    ######### ADDITIONS #################
    if original_data == "yes" or original_data == "y":
        DATA_FOLDER = "./original_data" 
    else:
        DATA_FOLDER = "./authors_data"
    #####################################

    dataset_path = os.path.join(DATA_FOLDER)
    print(os.path.join(dataset_path, "compas_data.npz"))
    f = np.load(os.path.join(dataset_path, "compas_data.npz"))

    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    Y_train[Y_train == 0] = -1
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    Y_test[Y_test == 0] = -1
    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

""" Added the conidition for the data folder and also added a parameter to the function """
def load_drug(original_data):
    ######### ADDED BY STUDENTS #########
    if original_data == "yes" or original_data == "y":
        DATA_FOLDER = "./original_data" 
    else:
        DATA_FOLDER = "./authors_data"
    #####################################

    dataset_path = os.path.join(DATA_FOLDER)
    print(os.path.join(dataset_path, "drug2_data.npz"))
    f = np.load(os.path.join(dataset_path, "drug2_data.npz"))

    # reshape and change the 0 values to -1 in the Y sets respectively
    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    Y_train[Y_train == 0] = -1
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    Y_test[Y_test == 0] = -1
    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def load_dataset(dataset_name, original_data, rand_seed):
    dataset_names = ["data.npz", "german_group_label.npz", "compas_data.npz", "compas_group_label.npz", 
                     "drug2_data.npz", "drug2_group_label.npz"]

    ################## ADDITIONS ##################################
    # Make sure to only make the dataset once (since they are making it only once as well)
    if original_data == "yes" or original_data == "y":
        if len([i for i in glob.glob(os.path.join("original_data", "*")) if i.split(os.path.join("original_data", " ").strip())[1] in dataset_names]) != len(dataset_names):
            make_datasets.create_orig_german_dataset()
            make_datasets.create_orig_compas_dataset()
            make_datasets.create_orig_drug_dataset()
    ###############################################################

    if dataset_name == 'german':
        return load_german(original_data)
    elif dataset_name == 'compas':
        return load_compas(original_data)
    elif dataset_name == 'drug':
        return load_drug(original_data)