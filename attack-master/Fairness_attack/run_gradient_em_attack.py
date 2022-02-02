from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
############################################ ADDITIONS ############################################
import csv
###################################################################################################
import argparse
import time

import numpy as np

import scipy.sparse as sparse
import data_utils as data
import datasets

import iterative_attack
from influence.influence.smooth_hinge import SmoothHinge
from influence.influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

import tensorflow as tf

############################################ ADDITIONS #################################################
time_start = time.time()
########################################################################################################

def get_projection_fn_for_dataset(dataset_name, X, Y, use_slab, use_LP, percentile):
    # check which dataset is used and use the get projection function
    if dataset_name in ['enron', 'imdb','german','compas','drug']:
        # get projection
        projection_fn = data.get_projection_fn(
            X, Y,
            sphere=True,
            slab=use_slab,
            non_negative=True,
            less_than_one=False,
            use_lp_rounding=use_LP,
            percentile=percentile)

    return projection_fn

# Params defined by Mehrabi et al.
fit_intercept = True
initial_learning_rate = 0.001
keep_probs = None
decay_epochs = [1000, 10000]
num_classes = 2
batch_size = 100
temp = 0.001
use_copy = True
use_LP = True

# The store true in the arguments used for storing values true and false respectively
parser = argparse.ArgumentParser()
parser.add_argument('--em_iter', default=1)
parser.add_argument('--total_grad_iter', default=300)
parser.add_argument('--use_slab', action='store_true')
parser.add_argument('--dataset', default='german')
parser.add_argument('--percentile', default=90)
parser.add_argument('--epsilon', default=0.03)
parser.add_argument('--step_size', default=0.1)
parser.add_argument('--use_train', action="store_true")
parser.add_argument('--baseline', action="store_true") # Means no LP, no copy, and no smooth
parser.add_argument('--baseline_smooth', action="store_true") # Means no LP, no copy
parser.add_argument('--no_LP', action="store_true")
parser.add_argument('--timed', action="store_true")
parser.add_argument('--sensitive_feature_idx', default=36)
parser.add_argument('--method', default="IAF")
parser.add_argument('--sensitive_attr_filename',default='german_group_label.npz')

############################################ ADDITIONS #####################################################
parser.add_argument('--recreated_data', default="no") # check if model runs with authors or recreated data
parser.add_argument('--rand_seed', default=0) # add given value to seeds in code (suggested 1, 2 or 3)
###########################################################################################################
args = parser.parse_args()

# Use all the arguments of the parser
dataset_name = args.dataset
use_slab = args.use_slab
epsilon = float(args.epsilon)
step_size = float(args.step_size)
percentile = int(np.round(float(args.percentile)))
max_em_iter = int(np.round(float(args.em_iter)))
total_grad_iter = int(np.round(float(args.total_grad_iter)))
use_train = args.use_train
baseline = args.baseline
baseline_smooth = args.baseline_smooth
no_LP = args.no_LP
timed = args.timed
attack_method = args.method
sensitive_idx = int(args.sensitive_feature_idx)
sensitive_file = args.sensitive_attr_filename

############################################ ADDITIONS #############################################
# check if recreated data is passed in and if there is an addition to the seeds
recreated_data = args.recreated_data
recreated_data = recreated_data.lower()
rand_seed = int(args.rand_seed)
np.random.seed(1+rand_seed)
####################################################################################################

# Check which attack method is used and define the loss type based on the attack method
if(attack_method == "IAF"):
    loss_type ='adversarial_loss'
else:
    loss_type = 'normal_loss'

print('epsilon: %s' % epsilon)
print('use_slab: %s' % use_slab)

# Check which dataset is used
if dataset_name == 'german':
    weight_decay = 0.09
elif dataset_name == 'compas':
    weight_decay = 0.09
elif dataset_name == 'drug':
    weight_decay = 0.09

# Means no LP, no copy, and no smooth
if baseline:
    temp = 0
    assert dataset_name == 'german'
    assert not baseline_smooth
    assert not use_train
    use_copy = False
    use_LP = False
    percentile = 80

# Means no LP, no copy, but smoothing is applied
if baseline_smooth:
    assert dataset_name == 'german'
    assert not baseline
    assert not use_train
    use_copy = False
    use_LP = False
    percentile = 80

# Means no LP
if no_LP:
    assert dataset_name == 'german'
    use_LP = False
    percentile = 80

# Make filename based on the used dataset, usage of slab, step size, epsilon and weight decay
model_name = 'smooth_hinge_%s_sphere-True_slab-%s_start-copy_lflip-True_step-%s_t-%s_eps-%s_wd-%s_rs-1' % (
                dataset_name, use_slab,
                step_size, temp, epsilon, weight_decay)

# Used percentile is added to file name
if percentile != 90:
    model_name = model_name + '_percentile-%s' % percentile
model_name += '_em-%s' % max_em_iter

# Used baselines/LP is added to file name
if baseline:
    model_name = model_name + '_baseline'
if baseline_smooth:
    model_name = model_name + '_baseline-smooth'
if no_LP:
    model_name = model_name + '_no-LP'
if timed:
    model_name = model_name + '_timed'

# add seed to folder name
if recreated_data == "yes" or recreated_data == "y":
    dataset_choice = "Recreated data" + " seed {}".format(rand_seed)
else:
    dataset_choice = "Authors data" + " seed {}".format(rand_seed)
    
# make path name to folder (results/ recreated or authors data/ dataset name)
dataset_namee = [i for i in model_name.split("_") if i in ["german", "drug", "compas"]][0] 

# Make folder if it does not exist
output_root = os.path.join(".", "{}".format("output"), "{}".format(dataset_choice), "{}".format(dataset_namee))

# Output_root = os.path.join(datasets.OUTPUT_FOLDER, dataset_name, 'influence_data')
datasets.safe_makedirs(output_root)

# Check if max_em_iter is 0, the default is 1 in the parser arguments. 
if max_em_iter == 0:
    num_grad_iter_per_em = total_grad_iter
else:
    assert total_grad_iter % max_em_iter == 0
    num_grad_iter_per_em = int(np.round(total_grad_iter / max_em_iter))

############################################ ADDITIONS ###############################################
# get the split of the dataset (which is already done in the npz files)
X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name, recreated_data, rand_seed)
######################################################################################################

# General train idx is the number of samples in X train
general_train_idx = X_train.shape[0]

# The unique sensitives are defined by getting the sorted unique values in the sensitive column
unique_sensitives = np.sort(np.unique(X_train[:,sensitive_idx]))

# Positive sensitives are defined as the second value in the unique senstives and negatives respectively as the first value
positive_sensitive_el = np.float32(unique_sensitives[1])
negative_sensitive_el = np.float32(unique_sensitives[0])

# Check if X train is sparse, if so, transform to array
if sparse.issparse(X_train):
    X_train = X_train.toarray()
if sparse.issparse(X_test):
    X_test = X_test.toarray()

# If argument use_train is true, use X and Y train as X test and Y test respectively
if use_train:
    X_test = X_train
    Y_test = Y_train

# Get class map, centroinds, centroid vector, spehere and slab radius based on function in data utils
class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
    X_train, Y_train, percentile=percentile)

# If the training samples times epsilon is smaller than 2, exit
if(X_train.shape[0]*epsilon < 2):
    print("The end")
    exit()

feasible_flipped_mask = iterative_attack.get_feasible_flipped_mask(
    X_train, Y_train,
    centroids,
    centroid_vec,
    sphere_radii,
    slab_radii,
    class_map,
    use_slab=use_slab)

# Modify x and y based on flipped mask
X_modified, Y_modified, indices_to_poison, copy_array, advantaged = iterative_attack.init_gradient_attack_from_mask(
    X_train, Y_train,
    epsilon,
    feasible_flipped_mask,
    general_train_idx,
    sensitive_file,
    attack_method,
    use_copy=use_copy,
    ####### ADDITIONS ################
    recreated_data=recreated_data,
    rand_seed=rand_seed
    ##################################
    )

tf.reset_default_graph()

# Make train, validation, test sets
input_dim = X_train.shape[1]
train = DataSet(X_train, Y_train)
validation = None
test = DataSet(X_test, Y_test)
data_sets = base.Datasets(train=train, validation=validation, test=test)

# Use smoothHinge (explained in file)
model = SmoothHinge(
    positive_sensitive_el = positive_sensitive_el,
    negative_sensitive_el = negative_sensitive_el,
    sensitive_feature_idx = sensitive_idx,
    input_dim=input_dim,
    temp=temp,
    weight_decay=weight_decay,
    use_bias=True,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    decay_epochs=None,
    mini_batch=False,
    train_dir=output_root,
    log_dir='log',
    model_name=model_name,
    method = attack_method,
    general_train_idx=general_train_idx,
    sensitive_file=sensitive_file,
    ####### ADDITIONS ################
    recreated_data=recreated_data,
    rand_seed=rand_seed
    ##################################
    )

# Update x and y and train
model.update_train_x_y(X_modified, Y_modified)
model.train()

# Check if timed argument is true, if so time it
if timed:
    start_time = time.time()
else:
    start_time = None

num_em_iters = max(max_em_iter, 1)

for em_iter in range(num_em_iters):

    print('\n\n##### EM iter %s #####' % em_iter)
    # Define for if max_em_iter is not 0
    X_modified = model.data_sets.train.x
    Y_modified = model.data_sets.train.labels

    if max_em_iter == 0:
        # Get projection
        projection_fn = get_projection_fn_for_dataset(
            dataset_name,
            X_train,
            Y_train,
            use_slab,
            use_LP,
            percentile)
    else:
        # Get projection
        projection_fn = get_projection_fn_for_dataset(
            dataset_name,
            X_modified,
            Y_modified,
            use_slab,
            use_LP,
            percentile)

    # The attacks (explained in functions and file)
    iterative_attack.iterative_attack(
        model,
        general_train_idx,
        sensitive_file,
        attack_method,
        advantaged,
        indices_to_poison=indices_to_poison,
        test_idx=None,
        test_description=None,
        step_size=step_size,
        num_iter=num_grad_iter_per_em,
        loss_type=loss_type,
        projection_fn=projection_fn,
        output_root=output_root,
        num_copies=copy_array,
        stop_after=2,
        start_time=start_time,
        ####### ADDITIONS ################
        recreated_data=recreated_data,
        rand_seed=rand_seed,
        model_name=model_name
        ##################################
        )

########################################## ADDITIONS #####################################################################
time_end = time.time()
total_time = {"time_taken_seconds": time_end - time_start}

""""Our extension to save the time taken """
if not os.path.isdir(os.path.join(".", "{}".format("results"))):
    os.mkdir(os.path.join(".", "{}".format("results")))

# add seed to folder name
if recreated_data == "yes" or recreated_data == "y":
    dataset_choice = "Recreated data" + " seed {}".format(rand_seed)
else:
    dataset_choice = "Authors data" + " seed {}".format(rand_seed)
    

if not os.path.isdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice))):
    os.mkdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice)))
    
# make path name to folder (results/ recreated or authors data/ dataset name)
dataset_namee = [i for i in model_name.split("_") if i in ["german", "drug", "compas"]][0] 

# Make folder if it does not yet exist
if not os.path.isdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice), "{}".format(dataset_namee))):
    os.mkdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice), "{}".format(dataset_namee)))

# Just a placeholder
time_and_it = "time_and_it"

last_path = os.path.join(".", "{}".format("results"), "{}".format(dataset_choice), "{}".format(dataset_namee), "{}".format(time_and_it))
# Make folder if it does not exist
if not os.path.isdir(last_path):
    os.mkdir(last_path)

# Save time taken in seconds for each epsilon in the concerning folder
csv_column = ['time_taken_seconds', 'iteration']
csv_file_name = '{}-{}.csv'.format(attack_method, model_name)
path_to_csv = os.path.join(last_path, csv_file_name)
if not os.path.isfile(path_to_csv):
    try:
        with open(path_to_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_column)
            writer.writeheader()

            writer.writerow(total_time)
    except IOError:
        print("I/O error")
else:
    with open(path_to_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_column)
        writer.writerow(total_time)

print("The end")
