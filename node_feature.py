import os
import argparse

from tqdm import tqdm
import h5py
import json
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary_file', type=str, help='dictionary file')
    parser.add_argument('--word_matrix_file', type=str, help='word embedding matrix file')
    parser.add_argument('--ids_map_folder', type=str, help='image index map file')
    parser.add_argument('--scene_graph_folder', type=str, help='file path save the scene graph')
    parser.add_argument('--export_folder', type=str, help='path to save result')
    parser.add_argument('--object_feature_folder', type=str, help='path where the object feature saved')
    return parser.parse_args()


class WordEmbedding:
    def __init__(self, dictionary_file, word_matrix_file):
        with open(dictionary_file, 'r') as f:
            self.word_to_idx = json.load(f)['word_to_ix']
        self.index_to_vector = np.load(word_matrix_file)

    def __call__(self, token):
        try:
            index = self.word_to_idx[token]
        except KeyError:
            index = self.word_to_idx['unknown']
        vector = self.index_to_vector[index]
        return vector


def image_node_feature_generation(word_embed, image_scene_graph, image_object_feature):
    object_name_embedding = np.zeros((36, 300), dtype='float32')
    for index, name in enumerate(image_scene_graph['objects'].values()):
        object_name_embedding[index] = word_embed(name)
    return np.concatenate((object_name_embedding, image_object_feature), dim=1)


def tier_node_feature_generation(tier, word_embed, args):
    print('#### Generating node feature for {} images ####'.format(tier))

    print('### Loading ids map...')
    with open(os.path.join(args.ids_map_folder, '{}_ids_map.json'.format(tier)), 'r') as f:
        image_id_to_ix = json.load(f)['image_id_to_ix']
    print('### ids map loaded!')

    print('### Loading scene graph...')
    with open(os.path.join(args.scene_graph_folder, '{}_sg.json'.format(tier)), 'r') as f:
        scene_graphs = json.load(f)
    print('### Scene graph loaded!')

    print('### Loading object features...')
    object_features = h5py.File(os.path.join(args.object_feature_folder, '{}_objects.h5'.format(tier)), 'r')
    print('### Object feature loaded!')

    node_feature_h5 = h5py.File(os.path.join(args.export_folder, '{}_node_feature'.format(tier)), 'w')
    node_feature_h5.create_dataset("node_feature", (len(image_id_to_ix), 36, 2348), dtype='f4')

    for image_id, index in tqdm(image_id_to_ix.items(), unit='image', desc='Node feature generation'):
        image_scene_graph = scene_graphs[image_id]
        image_object_feature = object_features['features'][index]
        image_node_feature = image_node_feature_generation(word_embed, image_scene_graph, image_object_feature)
        node_feature_h5["node_feature"][index] = image_node_feature

    node_feature_h5.close()


def main():
    args = get_args()

    word_embed = WordEmbedding(args.dictionary_file, args.word_matrix_file)

    for tier in ['train', 'val', 'test']:
        tier_node_feature_generation(tier, word_embed, args)


if __name__ == '__main__':
    main()

