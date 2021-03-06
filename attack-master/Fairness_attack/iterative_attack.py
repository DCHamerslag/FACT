from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import numpy as np

import os
import defenses
import data_utils as data
############################################ ADDED THIS ############################################
import csv
############################################ CHANGED THIS ############################################
import random

def poison_with_influence_proj_gradient_step(model, general_train_idx,
    sensitive_file,attack_method,advantaged, test_idx, indices_to_poison,
    projection_fn,
    step_size=0.01,
    shrink_towards='cluster_center',
    loss_type='normal_loss',
    force_refresh=True,
    test_description=None,
    output_root=None,
    ########### ADDITIONS ###################
    recreated_data=None,
    rand_seed=0
    #########################################
    ):
    """
    Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
    that has been modified by a single gradient step.
    """
    data_sets = model.data_sets

    # Define test description
    if test_description is None:
        test_description = test_idx
    
    # Save 
    grad_filename = os.path.join(output_root, 'grad_influence_wrt_input_val_%s_testidx_%s.npy' % (model.model_name, test_description))

    # If no refresh and the file already exists, load the existing file
    if (force_refresh == False) and (os.path.exists(grad_filename)):
        grad_influence_wrt_input_val = np.load(grad_filename)
    else:
        # Otherwise get gradient of the influence with respect to the input
        grad_influence_wrt_input_val = model.get_grad_of_influence_wrt_input(
            indices_to_poison,
            test_idx,
            verbose=False,
            force_refresh=force_refresh,
            test_description=test_description,
            loss_type=loss_type)

    # Poison X_train
    poisoned_X_train = data_sets.train.x[indices_to_poison, :]
    poisoned_X_train -= step_size * grad_influence_wrt_input_val

    # Poison labels
    poisoned_labels = data_sets.train.labels[indices_to_poison]

    weights = model.sess.run(model.weights)

    if(attack_method == "RAA"):
        ######### ADDITIONS #################
        if recreated_data == "yes" or recreated_data == "y":
            DATA_FOLDER = os.path.join(".", "recreated_data")
        else:
            DATA_FOLDER = os.path.join(".", "authors_data")
        #####################################

        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, sensitive_file))
        group_label = f['group_label']

        # Get indices for male and female
        male_train_index=np.where(group_label[0:general_train_idx] == 0)[0].astype(np.int32)
        female_train_index=np.where(group_label[0:general_train_idx] == 1)[0].astype(np.int32)

        # Make gender labels with 1 for male and -1 for female
        gender_labels = np.zeros(data_sets.train.labels.shape[0])
        for k in range(general_train_idx):
            if(k in male_train_index):
                gender_labels[k] = 1
            elif(k in female_train_index):
                gender_labels[k] = -1
        
        ############################################ ADDITIONS #######################################################
        random.seed(0+rand_seed)  # added + rand_seed (since if rand_seed = 0, it is the same as the original paper)
        ##############################################################################################################

        # Get op_idx based on advantaged group
        if(advantaged == -1):
            op_indx  = np.where((data_sets.train.labels == -1)& (gender_labels==-1))[0]
        else:
            op_indx  = np.where((data_sets.train.labels == -1)& (gender_labels==1))[0]

        # Poison a randomly chosen index
        rand1 = random.randint(0,op_indx.shape[0]-1)
        poisoned_X_train[0] = data_sets.train.x[op_indx[rand1], :]
 
        # Get op_idx based on advantaged group
        if(advantaged == -1):
            op_indx  = np.where((data_sets.train.labels == 1)& (gender_labels==1))[0]
        else:
            op_indx  = np.where((data_sets.train.labels == 1)& (gender_labels==-1))[0]

        # Poison a randomly chosen index
        rand2 = random.randint(0,op_indx.shape[0]-1)
        poisoned_X_train[1] = data_sets.train.x[op_indx[rand2], :]
 

    elif(attack_method == "NRAA"):
        ######### ADDITIONS ####################################
        if recreated_data == "yes" or recreated_data == "y":
            DATA_FOLDER = os.path.join(".", "recreated_data")
        else:
            DATA_FOLDER = os.path.join(".", "authors_data")
        ########################################################

        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, sensitive_file))
        group_label = f['group_label']

        # Get indices for male and female
        male_train_index=np.where(group_label[0:general_train_idx] == 0)[0].astype(np.int32)
        female_train_index=np.where(group_label[0:general_train_idx] == 1)[0].astype(np.int32)

        # Make gender labels with 1 for male and -1 for female
        gender_labels = np.zeros(data_sets.train.labels.shape[0])
        for k in range(general_train_idx):
            if(k in male_train_index):
                gender_labels[k] = 1
            elif(k in female_train_index):
                gender_labels[k] = -1

        ############################################ ADDITIONS #####################################################
        random.seed(0+rand_seed) # added + rand_seed (since if rand_seed = 0, it is the same as the original paper)
        ############################################################################################################
        
        # Get op_idx based on advantaged group
        if(advantaged == -1):
            op_indx  = np.where((data_sets.train.labels == -1)& (gender_labels==-1))[0]
        else:
            op_indx  = np.where((data_sets.train.labels == -1)& (gender_labels==1))[0]

        
        maxdist = 0
        maxpoint = 0
        for points in range(op_indx.shape[0]):
            temp = 0
            for p in range(op_indx.shape[0]):
                if(np.allclose(data_sets.train.x[op_indx[points], :],data_sets.train.x[op_indx[p], :],rtol=0,atol=1)):
                    temp = temp +1
            if(temp > maxdist):
                maxdist = temp
                maxpoint = points

        poisoned_X_train[0] = data_sets.train.x[op_indx[maxpoint], :]

        if(advantaged == -1):
            op_indx  = np.where((data_sets.train.labels == 1)& (gender_labels==1))[0]
        else:
            op_indx  = np.where((data_sets.train.labels == 1)& (gender_labels==-1))[0]

        maxdist = 0
        maxpoint = 0
        for points in range(op_indx.shape[0]):
            temp =0
            for p in range(op_indx.shape[0]):
                if(np.allclose(data_sets.train.x[op_indx[points], :],data_sets.train.x[op_indx[p], :],rtol=0,atol=3)):
                    temp = temp +1
            if(temp > maxdist):
                maxdist = temp
                maxpoint = points

        poisoned_X_train[1] = data_sets.train.x[op_indx[maxpoint], :]


    print('weights shape is ', weights.shape)
    poisoned_X_train = projection_fn(
        poisoned_X_train,
        poisoned_labels,
        theta=weights[:-1],
        bias=weights[-1])
 
    # Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
    # That has been modified by a single gradient step.
    return poisoned_X_train


def iterative_attack(
    model,
    general_train_idx,
    sensitive_file,
    attack_method,
    advantaged,
    indices_to_poison,
    test_idx,
    test_description=None,
    step_size=0.01,
    num_iter=10,
    loss_type='normal_loss',
    projection_fn=None,
    output_root=None,
    num_copies=None,
    stop_after=3,
    start_time=None,
    ########### ADDITIONS ##########
    recreated_data=None,
    rand_seed=0,
    model_name=None
    ################################
    ):

    # Sanity check
    if num_copies is not None:
        assert len(num_copies) == 2
        assert np.min(num_copies) >= 1
        assert len(indices_to_poison) == 2
        assert indices_to_poison[1] == (indices_to_poison[0] + 1)
        assert indices_to_poison[1] + num_copies[0] + num_copies[1] == (model.data_sets.train.x.shape[0] - 1)
        assert model.data_sets.train.labels[indices_to_poison[0]] == 1
        assert model.data_sets.train.labels[indices_to_poison[1]] == -1
        copy_start = indices_to_poison[1] + 1
        assert np.all(model.data_sets.train.labels[copy_start:copy_start+num_copies[0]] == 1)
        assert np.all(model.data_sets.train.labels[copy_start+num_copies[0]:copy_start+num_copies[0]+num_copies[1]] == -1)

    largest_test_loss = 0
    stop_counter = 0

    print('Test idx: %s' % test_idx)

    # Make zero matrices
    if start_time is not None:
        assert num_copies is not None
        times_taken = np.zeros(num_iter)
        Xs_poison = np.zeros((num_iter, len(indices_to_poison), model.data_sets.train.x.shape[1]))
        Ys_poison = np.zeros((num_iter, len(indices_to_poison)))
        nums_copies = np.zeros((num_iter, len(indices_to_poison)))

    for attack_iter in range(num_iter):
        print(num_iter)
        print('*** Iter: %s' % attack_iter)

        # Create modified training dataset
        old_poisoned_X_train = np.copy(model.data_sets.train.x[indices_to_poison, :])

        # Poisoned_X_train modified by a single gradient step
        poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
            model,
            general_train_idx,
            sensitive_file,
            attack_method,
            advantaged,
            test_idx,
            indices_to_poison,
            projection_fn,
            step_size=step_size,
            loss_type=loss_type,
            force_refresh=True,
            test_description=test_description,
            output_root=output_root,
            ############################################ ADDITIONS ############################################
            recreated_data=recreated_data,
            rand_seed=rand_seed
            ###################################################################################################
            )

        if num_copies is not None:
            poisoned_X_train = model.data_sets.train.x
            poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset
            copy_start = indices_to_poison[1] + 1
            poisoned_X_train[copy_start:copy_start+num_copies[0], :] = poisoned_X_train_subset[0, :]
            poisoned_X_train[copy_start+num_copies[0]:copy_start+num_copies[0]+num_copies[1], :] = poisoned_X_train_subset[1, :]
        else:
            poisoned_X_train = np.copy(model.data_sets.train.x)
            poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset

        # Calculate gradient step measures
        labels = model.data_sets.train.labels
        dists_sum = 0.0
        poisoned_dists_sum = 0.0
        poisoned_mask = np.array([False] * len(labels), dtype=bool)
        poisoned_mask[indices_to_poison] = True
        if(attack_method != "RAA" and attack_method!= "NRAA"):
            for y in set(labels):
                cluster_center = np.mean(poisoned_X_train[labels == y, :], axis=0)
                dists = np.linalg.norm(poisoned_X_train[labels == y, :] - cluster_center, axis=1)
                dists_sum += np.sum(dists)

                poisoned_dists = np.linalg.norm(poisoned_X_train[(labels == y) & (poisoned_mask), :] - cluster_center, axis=1)
                poisoned_dists_sum += np.sum(poisoned_dists)

            dists_mean = dists_sum / len(labels)
            poisoned_dists_mean = poisoned_dists_sum / len(indices_to_poison)

            dists_moved = np.linalg.norm(old_poisoned_X_train - poisoned_X_train[indices_to_poison, :], axis=1)
            print('Average distance to cluster center (overall): %s' % dists_mean)
            print('Average distance to cluster center (poisoned): %s' % poisoned_dists_mean)
            print('Average diff in X_train among poisoned indices = %s' % np.mean(dists_moved))
            print('Fraction of 0 gradient points: %s' % np.mean(dists_moved == 0))
            print('Average distance moved by points that moved: %s' % np.mean(dists_moved[dists_moved > 0]))

        # Update training dataset
        model.update_train_x(poisoned_X_train)

        # Retrain model
        results = model.train()

        if start_time is not None:
            end_time = time.time()
            times_taken[attack_iter] = end_time - start_time
            Xs_poison[attack_iter, :, :] = np.copy(poisoned_X_train_subset)
            Ys_poison[attack_iter, :] = model.data_sets.train.labels[indices_to_poison]
            nums_copies[attack_iter, :] = num_copies

        print('attack_iter', attack_iter)
        print('num_iter - 1', num_iter - 1)

        

        if ((attack_iter + 1) % 10 == 0) or (attack_iter == num_iter - 1):
            print('in')
            # Calculate test loss
            test_loss = results['test_loss']
            if largest_test_loss < test_loss:
                print('test loss match')
                largest_test_loss = test_loss
                np.savez(os.path.join(output_root, '%s_attack' % (model.model_name)),
                    poisoned_X_train=poisoned_X_train,
                    Y_train=model.data_sets.train.labels,
                    attack_iter=attack_iter + 1)

                stop_counter = 0
            else:
                stop_counter += 1
            
            
            if start_time is not None:
                np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
                    times_taken=times_taken,
                    nums_copies=nums_copies)
            if stop_counter >= stop_after:
                break
        ########################################## ADDITIONS #############################################################
        """"Our extension to save the time taken """
        iterations = {"iteration": attack_iter}

        if not os.path.isdir(os.path.join(".", "{}".format("results"))):
            os.mkdir(os.path.join(".", "{}".format("results")))

        if recreated_data == "yes" or recreated_data == "y":
            dataset_choice = "Recreated data" + " seed {}".format(rand_seed)
        else:
            dataset_choice = "Authors data" + " seed {}".format(rand_seed)

        if not os.path.isdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice))):
            os.mkdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice)))

        # make path name to folder (results/ recreated or authors data/ dataset name)
        dataset_namee = [i for i in model_name.split("_") if i in ["german", "drug", "compas"]][0] 

        # Make folder if it does not exist
        if not os.path.isdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice), "{}".format(dataset_namee))):
            os.mkdir(os.path.join(".", "{}".format("results"), "{}".format(dataset_choice), "{}".format(dataset_namee)))

        # Just a placeholder
        time_and_it = "time_and_it"

        last_path = os.path.join(".", "{}".format("results"), "{}".format(dataset_choice), "{}".format(dataset_namee), "{}".format(time_and_it))
        # Make folder if it does not yet exist
        if not os.path.isdir(last_path):
            os.mkdir(last_path)

        # Save number of iterations for each epsilon in the concerning folder
        csv_column = ['time_taken_seconds', 'iteration']
        csv_file_name = '{}-{}.csv'.format(attack_method, model_name)
        path_to_csv = os.path.join(last_path, csv_file_name)
        if not os.path.isfile(path_to_csv):
            try:
                with open(path_to_csv, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_column)
                    writer.writeheader()

                    writer.writerow(iterations)
            except IOError:
                print("I/O error")
        else:
            with open(path_to_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_column)
                writer.writerow(iterations)

        ########################################################################################################################

    # Save time measurements
    if start_time is not None:
        np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
            times_taken=times_taken,
            Xs_poison=Xs_poison,
            Ys_poison=Ys_poison,
            nums_copies=nums_copies)

def get_feasible_flipped_mask(
    X_train, Y_train,
    centroids,
    centroid_vec,
    sphere_radii,
    slab_radii,
    class_map,
    use_slab=False):

    # Computes ||Q(x - mu)|| in the corresponding norm.
    sphere_dists_flip = defenses.compute_dists_under_Q(
        X_train, -Y_train,
        Q=None,
        subtract_from_l2=False,
        centroids=centroids,
        class_map=class_map,
        norm=2)

    if use_slab:
        # Computes ||Q(x - mu)|| in the corresponding norm if slab is used
        slab_dists_flip = defenses.compute_dists_under_Q(
            X_train, -Y_train,
            Q=centroid_vec,
            subtract_from_l2=False,
            centroids=centroids,
            class_map=class_map,
            norm=2)

    # Create boolean vector of X_train's length
    feasible_flipped_mask = np.zeros(X_train.shape[0], dtype=bool)

    for y in set(Y_train):
        # For each class get flipped sphere radius 
        class_idx_flip = class_map[-y]
        sphere_radius_flip = sphere_radii[class_idx_flip]
        
        # Fill with true of false (0 or 1) based on condition
        feasible_flipped_mask[Y_train == y] = (sphere_dists_flip[Y_train == y] <= sphere_radius_flip)

        if use_slab:
            # If slab is used, more conditions are checked to fill with true or false (0 or 1)
            slab_radius_flip = slab_radii[class_idx_flip]
            feasible_flipped_mask[Y_train == y] = (
                feasible_flipped_mask[Y_train == y] &
                (slab_dists_flip[Y_train == y] <= slab_radius_flip))

    return feasible_flipped_mask


def init_gradient_attack_from_mask(
    X_train, Y_train,
    epsilon,
    feasible_flipped_mask,
    general_train_idx,
    sensitive_file,
    attack_method,
    use_copy=True,
    ########### ADDITIONS #############
    recreated_data=None,
    rand_seed=0
    ###################################
    ):

    
    ######### ADDITIONS #################
    if recreated_data == "yes" or recreated_data == "y":
        DATA_FOLDER = os.path.join(".", "recreated_data")
    else:
        DATA_FOLDER = os.path.join(".", "authors_data")
    #####################################

    # Get the group labels i.e. advantaged/disadvantaged 
    dataset_path = os.path.join(DATA_FOLDER)
    f = np.load(os.path.join(dataset_path, sensitive_file))
    group_label = f['group_label']

    # Defined advantaged as 1
    advantaged =1

    # Get all indices for male and female
    male_train_index=np.where(group_label[0:general_train_idx] == 0)[0].astype(np.int32)
    female_train_index=np.where(group_label[0:general_train_idx] == 1)[0].astype(np.int32)

    # Use logical and to check above condition together with value in Y train to get indices
    index_male_true_train = np.where(np.logical_and(group_label[0:general_train_idx] == 0, Y_train==1))[0].astype(np.int32)
    index_female_true_train = np.where(np.logical_and(group_label[0:general_train_idx] == 1, Y_train==1))[0].astype(np.int32)

    # Compute probabilites
    train_data_one_female_prob = group_label[0:general_train_idx][index_female_true_train].shape[0]/female_train_index.shape[0]
    train_data_one_male_prob = group_label[0:general_train_idx][index_male_true_train].shape[0]/male_train_index.shape[0]
   
   # Make gender labels with 1 and -1
    gender_labels = np.zeros(general_train_idx)
    for k in range(general_train_idx):
        if(k in male_train_index):
            gender_labels[k] = 1
        elif(k in female_train_index):
            gender_labels[k] = -1

    if not use_copy:
        # Calculate number of compies based on epsilon times samples in X_train
        num_copies = int(np.round(epsilon * X_train.shape[0]))

        # Randomly choose an idx to copy
        idx_to_copy = np.random.choice(
            np.where(feasible_flipped_mask)[0],
            size=num_copies,
            replace=True)

        # Concatenate the array which needs to be copied to X train
        X_modified = data.vstack(X_train, X_train[idx_to_copy, :])
        # Append flipped value which needs to be copied Y train
        Y_modified = np.append(Y_train, -Y_train[idx_to_copy])
        copy_array = None
        # Get all indices of the appended/concatenated values to poison 
        indices_to_poison = np.arange(X_train.shape[0], X_modified.shape[0])

    else:
        num_copies = int(np.round(epsilon * X_train.shape[0]))
        # Choose this in inverse class balance
        num_pos_copies = int(np.round(np.mean(Y_train == -1) * num_copies))
        num_neg_copies = num_copies - num_pos_copies

        ############################################ ADDITIONS #########################################################
        np.random.seed(0+rand_seed) # added + rand_seed (since if rand_seed = 0, it is the same as the original paper)
        ################################################################################################################

        # If female prob > male prob, advantaged is -1, since females are -1 (gender labels)
        if(train_data_one_female_prob>train_data_one_male_prob):
            advantaged = -1
            # Get positive and negative indices
            pos_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == 1)& (gender_labels==-1))[0])
            neg_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == -1)& (gender_labels==1))[0])
        else:
            # If male prob > female prob, advantaged is 1, since males are 1 (gender labels)
            advantaged = 1
            pos_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == 1)& (gender_labels==1))[0])
            neg_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == -1)& (gender_labels==-1))[0])

        # Checks to print the correct values for female and male
        if(neg_idx_to_copy in female_train_index):
            print("female")
        else:
            print("male")
        if(pos_idx_to_copy in female_train_index):
            print("female")
        else:
            print("male")
        print(neg_idx_to_copy)
        print(pos_idx_to_copy)

        # Subtract 1 from both pos and neg number of copies
        num_pos_copies -= 1
        num_neg_copies -= 1

        # Add all the points to x and y (see explanation in add points function)
        X_modified, Y_modified = data.add_points(
            X_train[pos_idx_to_copy, :],
            1,
            X_train,
            Y_train,
            num_copies=1)
        X_modified, Y_modified = data.add_points(
            X_train[neg_idx_to_copy, :],
            -1,
            X_modified,
            Y_modified,
            num_copies=1)
        X_modified, Y_modified = data.add_points(
            X_train[pos_idx_to_copy, :],
            1,
            X_modified,
            Y_modified,
            num_copies=num_pos_copies)
        X_modified, Y_modified = data.add_points(
            X_train[neg_idx_to_copy, :],
            -1,
            X_modified,
            Y_modified,
            num_copies=num_neg_copies)
        copy_array = [num_pos_copies, num_neg_copies]
        indices_to_poison = np.arange(X_train.shape[0], X_train.shape[0]+2)

    return X_modified, Y_modified, indices_to_poison, copy_array, advantaged
