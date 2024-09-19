import sys
import os
import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from typing import Dict, List, Tuple
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.svm_utils import get_split_sizes, get_svm_tr_test_idxs, get_softmax_dprime, get_svm_dprime

def load_model_results(directory_path: str, filename: str) -> Tuple[pd.DataFrame, List[float], List[float]]:
    full_path = os.path.join(directory_path, filename + '.pkl')
    with open(full_path, 'rb') as f:
        model_results, mean_accs, mean_losses = pickle.load(f)
    return pd.DataFrame(model_results), mean_accs, mean_losses

def main():
    # Parse command-line arguments
    directory_path = '/Users/davide/Documents/Work/github/model_training/results/model/'
    filename = sys.argv[1]
    N_SHUFFLE = int(sys.argv[2])

    # Load model results
    model_results, mean_accs, mean_losses = load_model_results(directory_path, filename)
    print('Model loaded:')
    print('Data stored as:', filename)

    # Prepare for SVM
    data_labs = model_results['category_num'].values
    valid_classes = model_results['category_num'].unique()
    N_FOLDS = 10
    N_TRAINING_IMGS = 50

    labs_indices, _ = get_split_sizes(valid_classes, data_labs, N_FOLDS)
    n_classes = len(valid_classes)
    which_activations = 'activations'
    DECIMALS = 5

    # Initialize dictionaries to store results
    all_ids = np.array(model_results['id'].tolist(), dtype=str)
    n_categories = len(model_results['category_num'].unique())
    decoding_results = {id_: np.full((n_categories, 2), np.nan) for id_ in all_ids}
    decoding_results_calibrated = {id_: np.full(n_categories, np.nan) for id_ in all_ids}

    # Get binary combinations of all categories
    category_combinations = np.array(list(combinations(range(n_categories), 2)))

    for shuffle_idx in range(N_SHUFFLE):
        print(f'Shuffle iteration: {shuffle_idx + 1}')
        train_idxs, test_idxs = get_svm_tr_test_idxs(labs_indices, N_TRAINING_IMGS, N_FOLDS, shuffle_idx, unbalanced=False)

        for fold_idx in range(N_FOLDS):
            print(f'    Fold: {fold_idx + 1}')
            train_idxs_fold = train_idxs[train_idxs['fold'] == fold_idx]
            test_idxs_fold = test_idxs[test_idxs['fold'] == fold_idx]

            for comb_idx in range(len(category_combinations)):
                cat1, cat2 = category_combinations[comb_idx]
                comb_train_sub = train_idxs_fold[train_idxs_fold['category'].isin([cat1, cat2])]
                comb_test_sub = test_idxs_fold[test_idxs_fold['category'].isin([cat1, cat2])]

                # Training phase
                train_activations = model_results[which_activations].iloc[comb_train_sub['index'].values]
                train_x = np.array(train_activations.tolist(), dtype=float)
                train_y = np.concatenate([np.zeros(len(comb_train_sub[comb_train_sub['category'] == cat1])),
                                        np.ones(len(comb_train_sub[comb_train_sub['category'] == cat2]))])

                classifier = LinearSVC(penalty='l2', loss='hinge', C=0.5, fit_intercept=True, max_iter=100000)
                classifier.fit(train_x, train_y)

                calibrated_clf = CalibratedClassifierCV(classifier, method='sigmoid')
                calibrated_clf.fit(train_x, train_y)

                # Testing phase
                test_activations = model_results[which_activations].iloc[comb_test_sub['index'].values]
                test_x = np.array(test_activations.tolist(), dtype=float)
                len_cat1 = len(comb_test_sub[comb_test_sub['category'] == cat1])
                len_cat2 = len(comb_test_sub[comb_test_sub['category'] == cat2])
                test_y = np.concatenate([np.zeros(len_cat1), np.ones(len_cat2)])

                predicted_labels = classifier.predict(test_x)
                probabilities_cal = calibrated_clf.predict_proba(test_x)

                # Calculate raw decision function scores
                raw_scores = np.dot(test_x, classifier.coef_.T) + classifier.intercept_
                probabilities = 1 / (1 + np.exp(-raw_scores))
                prob_cat1 = probabilities[:len_cat1]
                prob_cat2 = probabilities[-len_cat2:]
                cal_prob_cat1 = probabilities_cal[:len_cat1, 1]
                cal_prob_cat2 = probabilities_cal[-len_cat2:, 1]

                test_ids = np.array(model_results['id'].iloc[comb_test_sub['index']].tolist(), dtype=str)
                test_ids_cat1 = test_ids[:len_cat1]
                test_ids_cat2 = test_ids[-len_cat2:]

                # Update results for category 1
                for id1_idx, id1 in enumerate(test_ids_cat1):
                    prob = np.round(prob_cat1[id1_idx].item(), DECIMALS)
                    cal_prob = np.round(cal_prob_cat1[id1_idx].item(), DECIMALS)

                    # Update category 1 (target)
                    decoding_results[id1][cat2][0] = (decoding_results[id1][cat2][0] if not np.isnan(decoding_results[id1][cat2][0]) else 0) + (1 - prob)
                    decoding_results[id1][cat2][1] = (decoding_results[id1][cat2][1] if not np.isnan(decoding_results[id1][cat2][1]) else 0) + prob

                    # Update calibrated category 1
                    decoding_results_calibrated[id1][cat2] = (decoding_results_calibrated[id1][cat2] if not np.isnan(decoding_results_calibrated[id1][cat2]) else 0) + cal_prob

                # Update results for category 2
                for id2_idx, id2 in enumerate(test_ids_cat2):
                    prob = np.round(prob_cat2[id2_idx].item(), DECIMALS)
                    cal_prob = np.round(cal_prob_cat2[id2_idx].item(), DECIMALS)

                    # Update category 2 (target)
                    decoding_results[id2][cat1][0] = (decoding_results[id2][cat1][0].item() if not np.isnan(decoding_results[id2][cat1][0]) else 0) + prob
                    decoding_results[id2][cat1][1] = (decoding_results[id2][cat1][1].item() if not np.isnan(decoding_results[id2][cat1][1]) else 0) + (1 - prob)

                    # Update calibrated category 2
                    decoding_results_calibrated[id2][cat1] = (decoding_results_calibrated[id2][cat1] if not np.isnan(decoding_results_calibrated[id2][cat1]) else 0) + cal_prob


    # Average results
    decoding_results_avg = {id_: decoding_results[id_] / N_SHUFFLE for id_ in decoding_results}
    decoding_results_calibrated_avg = {id_: decoding_results_calibrated[id_] / N_SHUFFLE for id_ in decoding_results_calibrated}

    # Get softmax d-primes
    classes_to_int = {i:idx for idx,i in enumerate(model_results['category'].unique())}
    data = get_softmax_dprime(model_results, classes_to_int, DECIMALS)
    print('Softmax d-prime calculated')

    # Get other d-prime
    data = get_svm_dprime(data, decoding_results_avg, decoding_results_calibrated_avg, DECIMALS)
    print('SVM d-prime calculated')

    # Remove activation columns
    data = data.drop(columns=['logits', 'sigmoid', 'activations'])

    # Store results
    svm_path = '/Users/davide/Documents/Work/github/model_training/results/svm/'
    svm_store_path = os.path.join(svm_path, f'{filename}_svm.pkl')
    with open(svm_store_path, 'wb') as f:
        pickle.dump([data, mean_accs, mean_losses, decoding_results_avg, decoding_results_calibrated_avg], f)

    print('Data stored as:', svm_store_path)

if __name__ == '__main__':
    main()
