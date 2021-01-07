import argparse
import os

import h5py
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='train_val_test', help='train, val, test')
    parser.add_argument('--save_folder', type=str, help='path to save result')
    parser.add_argument('--node_feature_folder', type=str, help='path where save node features')
    parser.add_argument('--ids_map_folder', type=str, help='image index map file')

    return parser.parse_args()


def target_generation(h5_file_dir, ids_map_json, node_feature_dir):
    with open(ids_map_json, 'r') as f:
        image_ix_to_id = json.load(f)['image_ix_to_id']
    n_images = len(image_ix_to_id)
    node_feature = h5py.File(node_feature_dir, 'r')
    n_objects = node_feature['object_name_embedding'].shape[1]
    n_ocr = node_feature['ocr_bounding_boxes'].shape[1]
    n_nodes = n_objects + n_ocr
    target_h5 = h5py.File(h5_file_dir, 'w')
    target_h5.create_dataset("target", (n_images, n_nodes, 2), dtype='float32')
    target_h5['target'][:, n_objects:, 1] = 1
    target_h5.close()


def main():
    args = get_args()
    tiers = args.tiers.split('_')
    for tier in tiers:
        print('#### Generating target data for {} images ####'.format(tier))
        h5_file_dir = os.path.join(args.save_folder, '{}_target.h5'.format(tier))
        ids_map_json = os.path.join(args.ids_map_folder, '{}_ids_map.json'.format(tier))
        node_feature_dir = os.path.join(args.node_feature_folder, '{}_node_features.h5'.format(tier))
        target_generation(h5_file_dir, ids_map_json, node_feature_dir)


if __name__ == "__main__":
    main()