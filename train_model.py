import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.functional import F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import pickle
import numpy as np
import os
import sys
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_utils import (get_split_sizes, get_tr_test_idxs, get_loaders,
                   get_test_loaders, createNet, get_activation)


def main():
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Parameters from command line arguments
    which_cats = sys.argv[1]
    n_classes_to_use = sys.argv[2]
    n_shuff_iter = int(sys.argv[3])
    n_iterations = int(sys.argv[4])
    n_fold = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    FC_LAYER = 6 

    # Data preparation
    data_dir = f'/Users/davide/Documents/Work/github/model_training/data/images/{which_cats}/'

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=preprocess)
    labels = np.array(dataset.targets)
    class_counts = np.bincount(labels)

    # Filter classes based on minimum images per class
    min_images_per_class = 100
    valid_classes = np.where(class_counts >= min_images_per_class)[0]
    print(f"Valid classes found: {len(valid_classes)}")

    if n_classes_to_use == 'all':
        valid_classes = valid_classes 
    else:
        valid_classes = valid_classes[:int(n_classes_to_use)]

    indices_to_retain = [idx for idx, label in enumerate(labels) if label in valid_classes]
    filtered_dataset = Subset(dataset, indices_to_retain)

    # get mapping dict
    data_labs_nsd = [labels[i] for i in indices_to_retain] # is this useful?
    labs_map = {valid_classes[i]: i for i in range(len(valid_classes))}

    classes_str = dataset.classes
    valid_classes_str = [classes_str[i] for i in valid_classes]
    classes_str_map = {i:valid_classes_str[i] for i in range(len(valid_classes_str))}
    classes_to_int = {v: k for k, v in classes_str_map.items()}

    # get relevant variables
    data_labs = [labs_map[data_labs_nsd[i]] for i in range(len(data_labs_nsd))] # is this usefult too?
    data_ids = [dataset.imgs[idx][0].split('/')[-1].split('.')[0] for idx in indices_to_retain]
    data_info = {'id':data_ids, 'category':[classes_str_map[i] for i in data_labs]}

    # Initialize model and parameters
    dummy_model, _, _ = createNet(len(valid_classes), freeze=True)
    n_activations = int(dummy_model.classifier[FC_LAYER].out_features)

    # Flags
    fixed_train_size = True
    unbalanced = False
    use_minibatches = False

    store_path = (f'/Users/davide/Documents/Work/github/model_training/results/model/{which_cats}.pkl')
    print(f"Output file: {store_path}")

    # Initialize result storage
    test_acc = {id: [] for id in data_ids}
    test_acc_soft = {id: np.zeros(len(valid_classes)) for id in data_ids}
    test_reported_cat = {id: [] for id in data_ids}
    activations = {id: np.zeros(n_activations) for id in data_ids}
    losses = []
    accs = []
    test_acc_logit = {id: np.zeros(len(valid_classes)) for id in data_ids}

    # Main loop
    for shuff_iter in range(n_shuff_iter):
        print(f"\nShuffle iteration: {shuff_iter + 1}")
        
        if fixed_train_size:
            labs_indices, _ = get_split_sizes(valid_classes, data_labs, n_fold)
            n_train_imgs = 50
        else:
            labs_indices, n_train_imgs = get_split_sizes(valid_classes, data_labs, n_fold)

        train_idxs, test_idxs = get_tr_test_idxs(labs_indices, n_train_imgs, n_fold, seed=shuff_iter, unbalanced=unbalanced)
        test_loaders, train_loaders = get_loaders(filtered_dataset, train_idxs, test_idxs, n_fold, batch_size)
        
        fold_accs = []
        fold_losses = []

        for fold in range(n_fold):
            print(f"Cross-validation fold: {fold + 1}")
            
            test_ids = [data_ids[i] for i in test_idxs[fold]]
            model, optimizer, lossfun = createNet(len(valid_classes), freeze=True)
            model.train()

            running_loss = []
            running_acc = []
            
            train_dataset = Subset(filtered_dataset, train_idxs[fold])
            train_loader = (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2) 
                            if use_minibatches else 
                            DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=8, prefetch_factor=2))
            
            for n_iter in range(n_iterations):
                print(f"    Iteration number: {n_iter + 1}")
                
                for images, labels in train_loader:
                    X = images.to(device)
                    y = torch.tensor([labs_map[label.item()] for label in labels]).to(device)

                    yHat = model(X)
                    loss_train = lossfun(yHat, y)

                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                    running_loss.append(loss_train.item())
                    act_acc = (torch.mean((torch.argmax(yHat, dim=1) == y).float()).item()) * 100
                    running_acc.append(np.round(act_acc, 1))

            fold_losses.append(np.array(running_loss))
            fold_accs.append(np.array(running_acc))

            # Testing phase
            print("Testing")
            activation_fold = {}
            layer_to_extract = model.classifier[FC_LAYER]
            hook_function = get_activation('fc_8', activation_fold)
            layer_to_extract.register_forward_hook(hook_function)

            # Set to eval mode
            model.eval()
            test_images, test_labels = next(iter(test_loaders[fold]))

            test_X = test_images.to(device)
            y = torch.tensor([labs_map[label.item()] for label in test_labels]).to(device)

            with torch.no_grad():
                yHat = model(test_X)
                loss_test = lossfun(yHat, y)

            running_acc_bin = (torch.argmax(yHat, dim=1) == y).float().cpu().numpy()
            running_acc_soft = F.softmax(yHat.cpu(), dim=1).detach().numpy()
            running_acc_logit = yHat.cpu().detach().numpy()

            for idx, id in enumerate(test_ids):
                test_acc[id].append(running_acc_bin[idx])
                test_acc_soft[id] += running_acc_soft[idx]
                test_acc_logit[id] += running_acc_logit[idx]
                predicted_class_int = np.argmax(running_acc_soft, axis=1)
                predicted_class_str = [classes_str_map[int] for int in predicted_class_int]
                test_reported_cat[id].append(predicted_class_str[idx])
                act_activation = activation_fold['fc_8'][idx].cpu().numpy().round(4)
                activations[id] += act_activation

            # Cleanup
            del model, optimizer, lossfun
            torch.cuda.empty_cache()
            gc.collect()

        accs.append(fold_accs)
        losses.append(fold_losses)

    # Process and store results
    test_acc_mean = {k: np.mean(test_acc[k]) for k in test_acc.keys()}
    test_acc_soft_mean = {k: test_acc_soft[k] / n_shuff_iter for k in test_acc_soft.keys()}
    test_acc_logit_mean = {k: test_acc_logit[k] / n_shuff_iter for k in test_acc_logit.keys()}
    test_acc_sig_mean = {k: 1 / (1 + np.exp(-test_acc_logit_mean[k])) for k in test_acc_logit.keys()}
    mean_losses = [np.sum(arrays, axis=0) / n_shuff_iter for arrays in zip(*losses)]
    mean_accs = [np.sum(arrays, axis=0) / n_shuff_iter for arrays in zip(*accs)]
    for k in activations.keys():
        activations[k] /= n_shuff_iter

    category_num = [classes_to_int[cat] for cat in data_info['category']]
    model_results = {
        'id': list(test_acc_mean.keys()),
        'category': data_info['category'],
        'category_num': category_num,
        'acc': list(test_acc_mean.values()),
        'softmax': list(test_acc_soft_mean.values()),
        'responses': list(test_reported_cat.values()),
        'logits': list(test_acc_logit_mean.values()),
        'sigmoid': list(test_acc_sig_mean.values()),
        'activations': list(activations.values())
    }

    data_to_store = (model_results, mean_accs, mean_losses)

    with open(store_path, 'wb') as f:
        pickle.dump(data_to_store, f)

    print(f"Results stored at: {store_path}")


if __name__ == '__main__':
    main()