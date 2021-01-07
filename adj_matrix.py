import argparse
import os

import numpy as np
import h5py
import json
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='train_val_test', help='train, val, test')
    parser.add_argument('--save_folder', type=str, help='path to save result')
    parser.add_argument('--adj_matrix_folder', type=str, help='path where adj matrix saved')
    parser.add_argument('--ids_map_folder', type=str, help='image index map file')
    return parser.parse_args()


def find_max_len(adj_matrix):
    max_len = 0
    for item in adj_matrix.values():
        if max_len < len(item):
            max_len = len(item)
    return max_len


def generate_adj_matrix(h5_file_dir, adj_matrix_json, ids_map_json):
    with open(adj_matrix_json, 'r') as f:
        adj_matrix = json.load(f)
    with open(ids_map_json, 'r') as f:
        image_ix_to_id = json.load(f)['image_ix_to_id']
    max_len = find_max_len(adj_matrix)
    n_images = len(image_ix_to_id)
    adj_h5 = h5py.File(h5_file_dir, 'w')
    adj_h5.create_dataset("adjacent_matrix", (n_images, max_len, max_len), dtype='float32')

    for image_index in tqdm(range(len(image_ix_to_id)), unit='image'):
        image_id = image_ix_to_id[str(image_index)]
        image_adj_matrix = np.array(adj_matrix[image_id])
        image_adj_matrix_pad = np.zeros((max_len, max_len), dtype='float32')
        image_adj_matrix_pad[:image_adj_matrix.shape[0], :image_adj_matrix.shape[1]] = image_adj_matrix
        adj_h5['adjacent_matrix'][image_index] = image_adj_matrix_pad

    adj_h5.close()


def main():
    args = get_args()
    tiers = args.tiers.split('_')
    for tier in tiers:
        print('#### Generating adjacent matrix for {} images ####'.format(tier))
        h5_file_dir = os.path.join(args.save_folder, '{}_adj_matrix.h5'.format(tier))
        adj_matrix_json = os.path.join(args.adj_matrix_folder, '{}_edge_rdiou.json'.format(tier))
        ids_map_json = os.path.join(args.ids_map_folder, '{}_ids_map.json'.format(tier))
        generate_adj_matrix(h5_file_dir, adj_matrix_json, ids_map_json)


if __name__ == '__main__':
    main()
