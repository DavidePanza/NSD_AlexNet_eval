import os
import io
import requests
import pandas as pd
import pickle
import numpy as np
from collections import Counter

# =========================================
# NSD UTILITY FUNCTIONS
# =========================================

def get_most_common_cat(pd, act_cats, i):
    """
    Identifies the most common category from the annotations of a specific index in the DataFrame.

    Args:
        pd (pandas.DataFrame): DataFrame containing the annotations in a column named 'anns'.
        act_cats (list): List of possible categories to search for within the annotations.
        i (int): Index of the row in the DataFrame to retrieve annotations from.

    Returns:
        str or None: The most common category found in the annotations. Returns `None` if no categories are found.
    """
    # Concatenate all annotations into a single string and split into individual words
    act_anns = " ".join(pd.loc[i].anns).split()

    # Find categories that match any word in the annotations
    cat_in_ans = [cat for cat in act_cats if cat in act_anns]
    counter = Counter(cat_in_ans)

    if not counter:
        return None

    # Identify the most common category (handle ties by retaining the first occurrence)
    max_count = max(counter.values())
    most_common_elements = [key for key, count in counter.items() if count == max_count]

    for element in cat_in_ans:
        if element in most_common_elements:
            return element

    return None

def download_and_process_nsd_data(url, coco_pd):
    """
    Downloads the NSD data from the provided URL and merges it with the COCO data.

    Args:
        url (str): URL to the NSD dataset CSV file.
        coco_pd (pd.DataFrame): COCO dataset DataFrame to merge with the NSD data.

    Returns:
        pd.DataFrame: Merged DataFrame of NSD and COCO data.
        pd.DataFrame: DataFrame with single categories.
    """
    # Send HTTP request and check if successful
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download data. Status code: {response.status_code}")

    # Load CSV data into pandas DataFrame
    nsd_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    col_to_keep = ['cocoId', 'nsdId', 'cropBox']
    nsd_pd = nsd_df[col_to_keep].rename(columns={'cocoId': 'id'})
    
    # Merge with COCO data
    nsd_allCat = pd.merge(nsd_pd, coco_pd, on='id', how='left')

    return nsd_allCat


def assign_main_category(nsd_allCat):
    """
    Assigns the most common category (or single category) to the NSD data.

    Args:
        nsd_allCat (pd.DataFrame): Merged NSD and COCO DataFrame.

    Returns:
        pd.DataFrame: DataFrame with main categories assigned.
        pd.DataFrame: DataFrame with single categories.
    """
    main_cat = []
    single_cat = []

    # Iterate over each row to determine the main category
    for i in range(len(nsd_allCat)):
        act_cats = nsd_allCat.loc[i].cat_s
        if len(act_cats) > 1:
            most_common_cat = get_most_common_cat(nsd_allCat, act_cats, i)
            main_cat.append(most_common_cat if most_common_cat else np.nan)
        else:
            main_cat.extend(act_cats)
            single_cat.append(i)

    nsd_allCat['main_cat'] = main_cat
    nsd_singleCat = nsd_allCat.loc[single_cat].reset_index(drop=True)

    return nsd_allCat, nsd_singleCat


def save_dataframes_nsd(nsd_allCat, nsd_singleCat, data_path):
    """
    Saves the NSD dataframes to disk using pickle.

    Args:
        nsd_allCat (pd.DataFrame): DataFrame with all categories.
        nsd_singleCat (pd.DataFrame): DataFrame with single categories.
        data_path (str): Directory path to store the data.
    """
    with open(os.path.join(data_path, 'nsd_allCat.csv'), 'wb') as f:
        pickle.dump(nsd_allCat, f)
    with open(os.path.join(data_path, 'nsd_singleCat.csv'), 'wb') as f:
        pickle.dump(nsd_singleCat, f)


def load_dataframes_nsd(data_path):
    """
    Loads the NSD dataframes from disk using pickle.

    Args:
        data_path (str): Directory path where the data is stored.

    Returns:
        pd.DataFrame: DataFrame with all categories.
        pd.DataFrame: DataFrame with single categories.
    """
    with open(os.path.join(data_path, 'nsd_allCat.csv'), 'rb') as f:
        nsd_allCat = pickle.load(f)
    with open(os.path.join(data_path, 'nsd_singleCat.csv'), 'rb') as f:
        nsd_singleCat = pickle.load(f)
    
    return nsd_allCat, nsd_singleCat