import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torch.functional import F
from torchvision.models import AlexNet_Weights


def get_split_sizes(classes, labels, n_fold):
    """
    Get the sizes of training and testing splits for cross-validation.

    Args:
        classes (list): List of class labels.
        labels (np.ndarray): Array of labels for the dataset.
        n_fold (int): Number of folds for cross-validation.

    Returns:
        dict, int: A dictionary mapping each class to its indices, and the number of training images per class.
    """
    labs_indices = {}

    for cl in np.arange(len(classes)):
        act_idxs = np.where(labels == cl)
        labs_indices[cl] = act_idxs[0]

    smallest_cat = min(np.unique(np.array(labels), return_counts=True)[1])
    n_train_imgs = (smallest_cat - (np.ceil(smallest_cat / n_fold))).astype(int)

    return labs_indices, n_train_imgs


def get_tr_test_idxs(labs_indices, n_train_imgs, n_fold, seed, unbalanced=False):
    """
    Generate training and testing indices for cross-validation.

    Args:
        labs_indices (dict): Dictionary mapping classes to their indices.
        n_train_imgs (int): Number of training images per class.
        n_fold (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.
        unbalanced (bool): Whether to use an unbalanced dataset.

    Returns:
        dict, dict: Dictionaries of training and testing indices for each fold.
    """
    np.random.seed(seed)
    test_idxs = {i: [] for i in range(n_fold)}
    train_idxs = {i: [] for i in range(n_fold)}
    
    for i in labs_indices.keys():
        act_idxs = labs_indices[i].copy()
        np.random.shuffle(act_idxs)
        diff_tr_test = len(act_idxs) - n_train_imgs 
        n_test_imgs = np.ceil((n_train_imgs / n_fold) + (diff_tr_test / n_fold)).astype(int)
        for i in range(n_fold):
            sub_idxs = act_idxs[i * n_test_imgs: (i + 1) * n_test_imgs]
            test_idxs[i].extend(sub_idxs)
            if not unbalanced:
                tr_idxs = [i for i in act_idxs if i not in sub_idxs]
                np.random.shuffle(tr_idxs)
                sub_tr_idxs = tr_idxs[:n_train_imgs]
                train_idxs[i].extend(sub_tr_idxs)
            else:
                tr_idxs = [i for i in act_idxs if i not in sub_idxs]
                np.random.shuffle(tr_idxs)
                train_idxs[i].extend(tr_idxs)
    
    return train_idxs, test_idxs


def get_loaders(data, train_idxs, test_idxs, n_fold, batch_size):
    """
    Create DataLoader instances for training and testing sets.

    Args:
        data (Dataset): The dataset to load.
        train_idxs (dict): Dictionary of training indices for each fold.
        test_idxs (dict): Dictionary of testing indices for each fold.
        n_fold (int): Number of folds for cross-validation.
        batch_size (int): Batch size for DataLoader.

    Returns:
        list, list: Lists of DataLoader instances for testing and training sets.
    """
    train_loaders = []
    test_loaders = []

    for idx in range(n_fold):
        train_dataset = Subset(data, train_idxs[idx])
        test_dataset = Subset(data, test_idxs[idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=torch.cuda.device_count())
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=torch.cuda.device_count())

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return test_loaders, train_loaders


def get_test_loaders(data, test_idxs, n_fold):
    """
    Create DataLoader instances for testing sets.

    Args:
        data (Dataset): The dataset to load.
        test_idxs (dict): Dictionary of testing indices for each fold.
        n_fold (int): Number of folds for cross-validation.

    Returns:
        list: List of DataLoader instances for testing sets.
    """
    test_loaders = []
    
    for idx in range(n_fold):
        test_dataset = Subset(data, test_idxs[idx])
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        test_loaders.append(test_loader)
    
    return test_loaders


def createNet(num_classes, freeze=True, verbose=False):
    """
    Create and initialize the AlexNet model.

    Args:
        num_classes (int): Number of output classes.
        freeze (bool): Whether to freeze the feature extraction layers.
        verbose (bool): Whether to print the parameters being updated.

    Returns:
        nn.Module, optim.Optimizer, nn.Module: The model, optimizer, and loss function.
    """
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    set_parameter_requires_grad(model, freeze)

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    params_to_update = model.parameters()
    if verbose:
        print("Params to learn:")
        if freeze:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(params_to_update, lr=0.001, weight_decay=0.5)
    lossfun = nn.CrossEntropyLoss()

    return model, optimizer, lossfun


def get_activation(name, activation_fold):
    """
    Hook function to get activations from a specific layer.

    Args:
        name (str): The name of the layer.

    Returns:
        function: A hook function that stores activations.
    """
    def hook(model, input, output):
        activation_fold[name] = output.detach()
    return hook
