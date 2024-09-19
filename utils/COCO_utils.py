import os
import pickle
import pandas as pd
from pycocotools.coco import COCO

# =========================================
# COCO UTILITY FUNCTIONS
# =========================================

def initialize_coco(data_dir, data_type):
    """
    Initialize COCO API for the given dataset type (train or val).
    
    Args:
        data_dir (str): Directory where COCO images and annotations are stored.
        data_type (str): Type of dataset (e.g., 'train2017', 'val2017').
    
    Returns:
        tuple: A tuple containing initialized COCO objects for instances and captions.
    """
    ann_file = os.path.join(data_dir, f'annotations/instances_{data_type}.json')
    ann_file_caps = os.path.join(data_dir, f'annotations/captions_{data_type}.json')
    
    # Initialize COCO APIs
    coco = COCO(ann_file)
    coco_caps = COCO(ann_file_caps)
    
    return coco, coco_caps

def process_coco_data(coco, coco_caps, categories):
    """
    Process COCO dataset to extract image ids, categories, and captions.
    
    Args:
        coco (COCO): Initialized COCO instance object.
        coco_caps (COCO): Initialized COCO captions object.
        categories (dict): Mapping of COCO category numbers to names.
    
    Returns:
        tuple: Four lists containing category numbers, category names, image IDs, and captions.
    """
    coco_cat_n = []
    coco_cat_s = []
    coco_id = []
    coco_captions = []
    
    # Iterate through each category
    for cat in categories.values():
        cat_id = coco.getCatIds([cat])
        img_ids = coco.getImgIds(catIds=cat_id)
        
        coco_cat_n.extend(cat_id * len(img_ids))
        coco_cat_s.extend([cat] * len(img_ids))
        coco_id.extend(img_ids)
        
        # Get captions for each image
        for img_id in img_ids:
            ann_ids = coco_caps.getAnnIds(imgIds=img_id)
            anns = coco_caps.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            coco_captions.append(captions)
    
    return coco_cat_n, coco_cat_s, coco_id, coco_captions

def create_dataframe(coco_cat_n, coco_cat_s, coco_id, coco_captions, sub_number_map):
    """
    Create a pandas DataFrame from the COCO data.

    Args:
        coco_cat_n (list): List of COCO category numbers.
        coco_cat_s (list): List of COCO category names.
        coco_id (list): List of COCO image IDs.
        coco_captions (list): List of COCO captions.
        sub_number_map (dict): Mapping of COCO category numbers to adjusted category numbers.
    
    Returns:
        pd.DataFrame: DataFrame containing COCO image IDs, categories, and captions.
    """
    coco_pd_full = pd.DataFrame({
        'id': coco_id,
        'cat_n': coco_cat_n,
        'cat_s': coco_cat_s,
        'anns': coco_captions
    })
    
    # Adjust category numbers using the provided map
    coco_pd_full['cat_n'] = coco_pd_full['cat_n'].map(sub_number_map)
    
    return coco_pd_full

def get_unique_entries(coco_pd_full):
    """
    Filter out duplicate image entries and keep unique ones.

    Args:
        coco_pd_full (pd.DataFrame): DataFrame containing all COCO data.
    
    Returns:
        pd.DataFrame: DataFrame with unique COCO image entries.
    """
    unique_ids = coco_pd_full['id'].unique()
    
    coco_id_full = []
    coco_cat_n_full = []
    coco_cat_s_full = []
    coco_anns_full = []
    
    for u in unique_ids:
        subset = coco_pd_full[coco_pd_full['id'] == u]
        coco_id_full.append(u)
        coco_cat_n_full.append(subset['cat_n'].unique())
        coco_cat_s_full.append(subset['cat_s'].unique())
        coco_anns_full.append(subset['anns'].reset_index(drop=True)[0])
    
    return pd.DataFrame({
        'id': coco_id_full,
        'cat_n': coco_cat_n_full,
        'cat_s': coco_cat_s_full,
        'anns': coco_anns_full
    })

def save_dataframe_coco(coco_pd, store_path):
    """
    Save DataFrame to a file using pickle.

    Args:
        coco_pd (pd.DataFrame): DataFrame to be saved.
        store_path (str): Path where the DataFrame will be saved.
    """
    with open(store_path, 'wb') as f:
        pickle.dump(coco_pd, f)

def load_dataframe_coco(store_path):
    """
    Load a DataFrame from a pickle file.

    Args:
        store_path (str): Path where the DataFrame is stored.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    with open(store_path, 'rb') as f:
        return pickle.load(f)


