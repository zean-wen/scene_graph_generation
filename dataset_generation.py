import os
import argparse

import pickle
import h5py
import json
import numpy as np
from tqdm import tqdm

from config import Config
from utils import WordEmbedding


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='train_val_test', help='train, val, test')
    parser.add_argument('--save_dir', type=str, help='path to save result')
    parser.add_argument('--ids_map_dir', type=str, help='image index map file')
    parser.add_argument('--data_root', type=str, help='path where the input data saved')
    parser.add_argument('--glove_dictionary_file', type=str, help='dictionary file')
    parser.add_argument('--glove_word_matrix_file', type=str, help='word embedding matrix file')
    parser.add_argument('--fasttext_dictionary_file', type=str, help='dictionary file')
    parser.add_argument('--fasttext_word_matrix_file', type=str, help='word embedding matrix file')

    return parser.parse_args()


class NodeFeature:
    def __init__(self,
                 save_dir,
                 image_ix_to_id,
                 scene_graphs,
                 ocr,
                 visual_feature_h5,
                 word_emb_config):

        self.image_ix_to_id = image_ix_to_id
        self.scene_graphs = scene_graphs
        self.visual_feature_h5 = visual_feature_h5
        self.ocr = ocr
        self.n_images = len(self.image_ix_to_id)
        self.word_emb_config = word_emb_config

        node_feature_dir = os.path.join(save_dir, 'node_features')
        if not os.path.exists(node_feature_dir):
            os.mkdir(node_feature_dir)
        self.dir = {'object_name_embeddings': os.path.join(node_feature_dir, 'object_name_embeddings'),
                    'object_visual_features': os.path.join(node_feature_dir, 'object_visual_features'),
                    'ocr_token_embeddings': os.path.join(node_feature_dir, 'ocr_token_embeddings'),
                    'ocr_bounding_boxes': os.path.join(node_feature_dir, 'ocr_bounding_boxes')}
        for path in self.dir.values():
            if not os.path.exists(path):
                os.mkdir(path)

    def generate(self):
        self.object_name_embedding_generation()
        self.object_visual_feature_generation()
        self.ocr_feature_generation()

    def object_name_embedding_generation(self):
        word_embed = WordEmbedding('glove', self.word_emb_config)
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Object name embedding generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_scene_graph = self.scene_graphs[image_id]
            n_objects = len(image_scene_graph['objects'])
            image_object_name_embeddings = np.zeros((n_objects, 300), dtype='float32')
            for object_index, _object in enumerate(image_scene_graph['objects'].values()):
                image_object_name_embeddings[object_index] = word_embed(_object['name'])
            with open(os.path.join(self.dir['object_name_embeddings'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_object_name_embeddings, f)

    def object_visual_feature_generation(self):
        object_visual_features = h5py.File(self.visual_feature_h5, 'r')
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Object visual feature generation'):
            image_object_visual_features = object_visual_features['features'][image_index]
            # remove rows with only zero elements
            image_object_visual_features = image_object_visual_features[~np.all(
                image_object_visual_features == 0,
                axis=1)]
            with open(os.path.join(self.dir['object_visual_features'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_object_visual_features, f)
        object_visual_features.close()

    def ocr_feature_generation(self):
        word_embed = WordEmbedding('fasttext', self.word_emb_config)

        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Ocr feature generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_ocr = self.ocr[image_id]
            n_ocr = len(image_ocr)
            image_ocr_token_embeddings = np.zeros((n_ocr, 300), dtype='float32')
            image_ocr_bounding_boxes = np.zeros((n_ocr, 8), dtype='float32')
            for ocr_index, (ocr_token, bbox) in enumerate(image_ocr.items()):
                image_ocr_token_embeddings[ocr_index] = word_embed(ocr_token)
                image_ocr_bounding_boxes[ocr_index] = np.array(bbox).flatten()
            with open(os.path.join(self.dir['ocr_token_embeddings'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_ocr_token_embeddings, f)
            with open(os.path.join(self.dir['ocr_bounding_boxes'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_ocr_bounding_boxes, f)


class AdjMatrix:
    def __init__(self, save_dir, image_ix_to_id, adj_matrix_dir):
        self.save_dir = os.path.join(save_dir, 'adjacent_matrix')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.adj_matrix_json = adj_matrix_dir
        self.image_ix_to_id = image_ix_to_id

    def generate(self):
        with open(self.adj_matrix_json, 'r') as f:
            adjacent_matrix = json.load(f)

        n_images = len(self.image_ix_to_id)
        for image_index in tqdm(range(n_images),
                                unit='image',
                                desc='Adjacent matrix generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_adj_matrix = np.array(adjacent_matrix[image_id])
            with open(os.path.join(self.save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_adj_matrix, f)


class Target:
    def __init__(self, save_dir, image_ix_to_id, scene_graphs, ocr):
        self.save_dir = os.path.join(save_dir, 'targets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.image_ix_to_id = image_ix_to_id
        self.ocr = ocr
        self.scene_graphs = scene_graphs

    def generate(self):
        n_images = len(self.image_ix_to_id)
        for image_index in tqdm(range(n_images),
                                unit='image',
                                desc='target generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            n_ocr = len(self.ocr[image_id])
            image_scene_graph = self.scene_graphs[image_id]
            n_objects = len(image_scene_graph['objects'])
            n_nodes = n_objects + n_ocr
            image_target = np.zeros(n_nodes, 2)
            image_target[:n_objects, 0] = 1
            image_target[n_objects:, 1] = 1
            with open(os.path.join(self.save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_target, f)


class DataSet:
    def __init__(self,
                 tier,
                 save_dir,
                 ids_map_dir,
                 data_root,
                 word_emb_config):
        self.tier = tier
        save_dir = os.path.join(save_dir, 'textvqa_{}'.format(tier))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        ids_map_dir = os.path.join(ids_map_dir, '{}_ids_map.json'.format(tier))
        scene_graph_dir = os.path.join(data_root, 'scene_graph', '{}_sg.json'.format(tier))
        visual_feature_dir = os.path.join(data_root, 'object_visual_feature', '{}_objects.h5'.format(tier))
        if tier == 'val':
            ocr_dir = os.path.join(data_root, 'ocr', '{}_ocr.json'.format('train'))
        else:
            ocr_dir = os.path.join(data_root, 'ocr', '{}_ocr.json'.format(tier))
        adj_matrix_dir = os.path.join(data_root, 'adjacent_matrix', '{}_edge_rdiou.json'.format(tier))

        with open(ids_map_dir, 'r') as f:
            image_ix_to_id = json.load(f)['image_ix_to_id']
        with open(ocr_dir, 'r') as f:
            ocr = json.load(f)
        with open(scene_graph_dir, 'r') as f:
            scene_graphs = json.load(f)

        self.node_feature = NodeFeature(save_dir,
                                        image_ix_to_id,
                                        scene_graphs,
                                        ocr,
                                        visual_feature_dir,
                                        word_emb_config)
        self.adjacent_matrix = AdjMatrix(save_dir, image_ix_to_id, adj_matrix_dir)
        self.target = Target(save_dir, image_ix_to_id, scene_graphs, ocr)

    def generate(self):
        print('#### Generating graph data for {} images ####'.format(self.tier))
        self.node_feature.generate()
        self.adjacent_matrix.generate()
        self.target.generate()


def main():
    args = get_args()
    config = Config.copy_from_args(args)

    tiers = config.tiers.split('_')
    for tier in tiers:
        data_set = DataSet(tier,
                           config.save_dir,
                           config.ids_map_dir,
                           config.data_root,
                           config.word_emb_config)
        data_set.generate()


if __name__ == '__main__':
    main()
