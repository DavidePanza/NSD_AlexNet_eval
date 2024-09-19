import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

def get_split_sizes(classes: list[int], labels: np.ndarray, n_fold: int) -> Tuple[Dict[int, np.ndarray], int]:
    """
    Compute training split sizes and indices for cross-validation.

    Args:
        classes (list[int]): List of class labels.
        labels (np.ndarray): Array of dataset labels.
        n_fold (int): Number of folds for cross-validation.

    Returns:
        Tuple[Dict[int, np.ndarray], int]: Dictionary of class indices and the number of training images per class.
    """
    labs_indices = {}
    num_classes = len(classes) if not isinstance(classes, int) else classes

    for cl in range(num_classes):
        labs_indices[cl] = np.where(labels == cl)[0]

    smallest_cat_size = min(np.bincount(labels))
    n_train_imgs = int(smallest_cat_size - np.ceil(smallest_cat_size / n_fold))

    return labs_indices, n_train_imgs

def get_svm_tr_test_idxs(labs_indices: Dict[int, np.ndarray], n_train_imgs: int, n_fold: int, seed: int, unbalanced: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate training and testing indices for cross-validation.

    Args:
        labs_indices (Dict[int, np.ndarray]): Dictionary of class indices.
        n_train_imgs (int): Number of training images per class.
        n_fold (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.
        unbalanced (bool): Whether to use an unbalanced dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of training and testing indices for each fold.
    """
    np.random.seed(seed)
    
    test_idxs = {'index': [], 'fold': [], 'category': []}
    train_idxs = {'index': [], 'fold': [], 'category': []}

    for cat, indices in labs_indices.items():
        np.random.shuffle(indices)
        diff_tr_test = len(indices) - n_train_imgs
        n_test_imgs = int(np.ceil((n_train_imgs / n_fold) + (diff_tr_test / n_fold)))
        
        for fold in range(n_fold):
            start_idx = fold * n_test_imgs
            end_idx = (fold + 1) * n_test_imgs
            sub_idxs = indices[start_idx:end_idx]
            test_idxs['index'].extend(sub_idxs)
            test_idxs['fold'].extend([fold] * len(sub_idxs))
            test_idxs['category'].extend([cat] * len(sub_idxs))
            
            if not unbalanced:
                tr_idxs = [idx for idx in indices if idx not in sub_idxs]
                np.random.shuffle(tr_idxs)
                sub_tr_idxs = tr_idxs[:n_train_imgs]
                train_idxs['index'].extend(sub_tr_idxs)
                train_idxs['fold'].extend([fold] * len(sub_tr_idxs))
                train_idxs['category'].extend([cat] * len(sub_tr_idxs))
            else:
                tr_idxs = [idx for idx in indices if idx not in sub_idxs]
                np.random.shuffle(tr_idxs)
                train_idxs['index'].extend(tr_idxs)
                train_idxs['fold'].extend([fold] * len(tr_idxs))
                train_idxs['category'].extend([cat] * len(tr_idxs))

    return pd.DataFrame(train_idxs), pd.DataFrame(test_idxs)

def get_softmax_dprime(data: pd.DataFrame, classes_to_int: Dict[str, int], decimals: int) -> pd.DataFrame:
    """
    Compute d-prime metrics from softmax outputs.

    Args:
        data (pd.DataFrame): DataFrame containing 'softmax' and 'category' columns.
        classes_to_int (Dict[str, int]): Mapping of class names to integer indices.
        decimals (int): Number of decimal places to round results.

    Returns:
        pd.DataFrame: Updated DataFrame with 'hitrate', 'FA', and 'd_prime' columns.
    """
    n_img = len(data)
    all_cats = data['category'].unique()
    
    hit_rate = []
    for img_idx in range(n_img):
        act_softmax = data.at[img_idx, 'softmax']
        act_cat = data.at[img_idx, 'category']
        act_cat_soft_idx = classes_to_int[act_cat]
        target_cats = [x for x in all_cats if x != act_cat]
        tmp_hit = 0

        for tgt in target_cats:
            tgt_soft_idx = classes_to_int[tgt]
            tmp_hit += act_softmax[act_cat_soft_idx] / (act_softmax[act_cat_soft_idx] + act_softmax[tgt_soft_idx])
        
        act_hit = np.round(tmp_hit / (len(all_cats) - 1), decimals)
        act_hit = np.round(1 - 1 / (2 * 340), decimals) if act_hit == 1 else act_hit
        hit_rate.append(act_hit)

    data['hitrate'] = hit_rate

    FA = []
    for img_idx in range(n_img):
        act_cat = data.at[img_idx, 'category']
        distractors_idx = data['category'] != act_cat
        tmp_hitrate = np.mean(data.loc[distractors_idx, 'hitrate'])
        act_FA = np.round(1 - tmp_hitrate, decimals)
        FA.append(act_FA)

    data['FA'] = FA
    data['d_prime'] = stats.norm.ppf(data['hitrate']) - stats.norm.ppf(data['FA'])

    return data

def get_svm_dprime(data: pd.DataFrame, decoding_results_avg: Dict[int, np.ndarray], decoding_results_calibrated_avg: Dict[int, np.ndarray], decimals: int) -> pd.DataFrame:
    """
    Compute d-prime metrics from SVM hit rates.

    Args:
        data (pd.DataFrame): DataFrame containing 'id' and 'category' columns.
        decoding_results_avg (Dict[int, np.ndarray]): Decoding results before calibration.
        decoding_results_calibrated_avg (Dict[int, np.ndarray]): Decoding results after calibration.
        decimals (int): Number of decimal places to round results.

    Returns:
        pd.DataFrame: Updated DataFrame with 'hit_rate_activations', 'hit_rate_activations_cal', 'FA_activations', 'FA_activations_cal', 'd_prime_activations', and 'd_prime_activations_cal' columns.
    """
    hit_rate_activations = []
    hit_rate_activations_cal = []

    for img_idx in range(len(data)):
        act_id = data.at[img_idx, 'id']
        act_hit = np.round(np.nanmean(decoding_results_avg[act_id][:, 0]), decimals)
        act_hit_cal = np.round(np.nanmean(decoding_results_calibrated_avg[act_id][:]), decimals)
        hit_rate_activations.append(act_hit)
        hit_rate_activations_cal.append(act_hit_cal)

    data['hit_rate_activations'] = hit_rate_activations
    data['hit_rate_activations_cal'] = hit_rate_activations_cal

    FA_activations = []
    FA_activations_cal = []

    for img_idx in range(len(data)):
        act_cat = data.at[img_idx, 'category']
        distractors_idx = data['category'] != act_cat
        tmp_hitrate = np.mean(data.loc[distractors_idx, 'hit_rate_activations'])
        tmp_hitrate_cal = np.mean(data.loc[distractors_idx, 'hit_rate_activations_cal'])
        FA_activations.append(np.round(1 - tmp_hitrate, decimals))
        FA_activations_cal.append(np.round(1 - tmp_hitrate_cal, decimals))

    data['FA_activations'] = FA_activations
    data['FA_activations_cal'] = FA_activations_cal
    data['d_prime_activations'] = stats.norm.ppf(data['hit_rate_activations']) - stats.norm.ppf(data['FA_activations'])
    data['d_prime_activations_cal'] = stats.norm.ppf(data['hit_rate_activations_cal']) - stats.norm.ppf(data['FA_activations_cal'])

    return data