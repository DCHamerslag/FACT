"""
We made this file to create the datasets. By Reversed engineering, we knew how the datasets were made. 
Below are the functions to create the datasets. These are called when the 'authors_data != yes' parameter.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

########################### GERMAN DATASET ###########################
def create_orig_german_dataset(rand_seed):
    np.random.seed(0+rand_seed)
    file_path = "original_data/resources/german.data"
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    
    targets = data[data.columns[-1]] # TARGET labels
    data = data.drop(20, axis=1) # drop targets before rescaling

    ## had to change the targets since the targets were [1,2]
    targets = targets.replace({1:0, 2:1})


    """
    Attribute 9 (in our dataset attribute and index 8, since we start at 0, which later becomes idx 0):
    Personal status and sex 
    A91 : male : divorced/separated 
    A92 : female : divorced/separated/married 
    A93 : male : single 
    A94 : male : married/widowed 
    A95 : female : single 
    """

    ## Sex attribute binary
    data[8] = data[8].replace({"A91": 0, "A92": 1, "A93": 0, "A94": 0, "A95":1})

    ## Sensitive feature is sex - attribute 8, make that now index 0
    sensitive_feature_idx = data.pop(8)
    data.insert(0, 8, sensitive_feature_idx)
    data = data.rename(columns={i:j for i,j in zip(data.columns, range(13))})

    # One-hot encode all categorical variables
    str_columns = []
    not_str = []
    for i in data.columns:
        if type(data[i][0]) == str:
            str_columns.append(i)
        else:
            not_str.append(i)


    dummies = pd.get_dummies(data[str_columns])
    data = pd.concat([data[not_str], dummies], axis=1, join='inner') 

    # First rescale to mean = 0 and std = 1, before adding targets to df (otherwise targets would be rescaled as well)
    for i in data.columns:
        data[i] = preprocessing.scale(data[i])

    dataset = pd.concat([data, targets], axis=1, join='inner')

    # Thereafter reshuffle whole dataframe 
    dataset = dataset.sample(frac=1, random_state=1+rand_seed).reset_index(drop=True)
    group_label = dataset.iloc[:, -1:].to_numpy()
    group_label = np.array([i[0] for i in group_label])

    # Split dataframe in 80-20%
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # At last make x and y
    X_train = train.iloc[:, :-1].to_numpy() # exclude targets
    X_test = test.iloc[:, :-1].to_numpy()

    y_train = train.iloc[:, -1:].to_numpy() # targets only
    y_train = np.array([i[0] for i in y_train])

    y_test = test.iloc[:, -1:].to_numpy() # targets only
    y_test = np.array([i[0] for i in y_test])

    # Just a check
    # print(len(X_train), len(X_test), len(y_train), len(y_test), len(group_label) == len(y_train) + len(y_test))
    
    np.savez("original_data/data.npz", X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
    np.savez("original_data/german_group_label.npz", group_label=group_label)
######################################################################


########################### COMPAS DATASET ###########################
def create_orig_compas_dataset(rand_seed):
    np.random.seed(0+rand_seed)

    data = pd.read_csv("original_data/resources/compas-scores-two-years.csv")
    targets = data[data.columns[-1]]

    # Used columns as specified in the paper
    used_cols = ["sex", "juv_fel_count", "priors_count", "race", "age_cat", 
                "juv_misd_count", "c_charge_degree", "juv_other_count"]

    data = data[used_cols]
    # Manually change the values male to 0 and female to 1
    data["sex"] = data["sex"].replace({"Male":0, "Female":1})
    str_columns = [i for i in data.columns if type(data[i][0]) == str]
    not_str = [i for i in data.columns if type(data[i][0]) != str]
    dummies = pd.get_dummies(data[str_columns])
    data = pd.concat([data[not_str], dummies], axis=1, join='inner') 

    # First rescale to mean = 0 and std = 1, before adding targets to df (otherwise targets would be rescaled as well)
    for i in data.columns:
        data[i] = preprocessing.scale(data[i])

    # print("Column specifications (as on website):", [i for i in data.columns])

    dataset = pd.concat([data, targets], axis=1, join='inner')

    # Thereafter reshuffle whole dataframe 
    dataset = dataset.sample(frac=1, random_state=1+rand_seed).reset_index(drop=True)
    group_label = dataset.iloc[:, -1:].to_numpy()
    group_label = np.array([i[0] for i in group_label])

    # Split dataframe in 80-20%
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # At last make x and y
    X_train = train.iloc[:, :-1].to_numpy() # exclude targets
    X_test = test.iloc[:, :-1].to_numpy()

    y_train = train.iloc[:, -1:].to_numpy() # targets only
    y_train = np.array([i[0] for i in y_train])

    y_test = test.iloc[:, -1:].to_numpy() # targets only
    y_test = np.array([i[0] for i in y_test])

    # Just a check
    # print(len(X_train), len(X_test), len(y_train), len(y_test), len(group_label) == len(y_train) + len(y_test))

    np.savez("original_data/compas_data.npz", X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
    np.savez("original_data/compas_group_label.npz", group_label=group_label)
######################################################################


########################### DRUG DATASET ###########################
def create_orig_drug_dataset(rand_seed):
    np.random.seed(0+rand_seed)
    
    file_path = "original_data/resources/drug_consumption.data"
    data = pd.read_csv(file_path, delimiter=",", header=None)

    targets = data.iloc[:, 20] ### Targets. In the real dataset it is attribute 21 (python goes from 0, thus 20 in our case).
    data = data.iloc[:, :13] ### They only take the first 13 attributes. See below the column specifications.

    """
    Column specifications 
    0 = Gender, 1 = ID, 2 = Age, 3 = Education, 4 = Country, 5 = Ethinicity, 6 = NScore, 7 = EScore,
    8 = OScore, 9 = AScore, 10 = CScore, 11 = Impulsiveness, 12 = SS, 20 = TARGET
    """

    ## Sensitive feature is gender - attribute 3, make that now index 0
    sensitive_feature_idx = data.pop(2)
    data.insert(0, 2, sensitive_feature_idx)
    data = data.rename(columns={i:j for i,j in zip(data.columns, range(13))})

    """
    Problem which can be solved:
    * Seven class classifications for each drug separately.
    * Problem can be transformed to binary classification by union of part of classes into one new class. 
    For example, "Never Used", "Used over a Decade Ago" form class "Non-user" and 
    all other classes form class "User".
    """

    """
    CL0 Never Used
    CL1 Used over a Decade Ago 
    CL2 Used in Last Decade 
    CL3 Used in Last Year 
    CL4 Used in Last Month 
    CL5 Used in Last Week 
    CL6 Used in Last Day 
    """

    targets = targets.replace({"CL0":0, "CL1":1, "CL2":1, "CL3":1, "CL4":1, "CL5":1, "CL6":1})

    # First rescale to mean = 0 and std = 1, before adding targets to df (otherwise targets would be rescaled as well)
    for i in data.columns:
        data[i] = preprocessing.scale(data[i])

    dataset = pd.concat([data, targets], axis=1, join='inner') 

    # Thereafter reshuffle whole dataframe 
    dataset = dataset.sample(frac=1, random_state=1+rand_seed).reset_index(drop=True)
    group_label = dataset.iloc[:, -1:].to_numpy()
    group_label = np.array([i[0] for i in group_label])

    # Split dataframe in 80-20%
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # At last make x and y
    X_train = train.iloc[:, :-1].to_numpy() # exclude targets
    X_test = test.iloc[:, :-1].to_numpy()

    y_train = train.iloc[:, -1:].to_numpy() # targets only
    y_train = np.array([i[0] for i in y_train])

    y_test = test.iloc[:, -1:].to_numpy() # targets only
    y_test = np.array([i[0] for i in y_test])

    # Just a check
    # print(len(X_train), len(X_test), len(y_train), len(y_test), len(group_label) == len(y_train) + len(y_test))

    np.savez("original_data/drug2_data.npz", X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
    np.savez("original_data/drug2_group_label.npz", group_label=group_label)
######################################################################