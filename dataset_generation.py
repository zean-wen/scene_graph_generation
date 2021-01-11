import os
import argparse

import pickle
import h5py
import numpy as np
from tqdm import tqdm

from config import Config
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='train_val_test', help='train, val, test')
    parser.add_argument('--save_dir', type=str, help='path to save result')
    parser.add_argument('--data_root', type=str, help='path where the input data saved')

    return parser.parse_args()


class NodeFeature:
    def __init__(self,
                 save_dir,
                 image_ix_to_id,
                 n_images,
                 nodes,
                 image_n_objects,
                 ocr,
                 visual_feature_h5,
                 word_emb_config):

        self.image_ix_to_id = image_ix_to_id
        self.nodes = nodes
        self.image_n_objects = image_n_objects
        self.visual_feature_h5 = visual_feature_h5
        self.ocr = ocr
        self.n_images = n_images
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
            image_nodes = self.nodes[image_id]
            n_objects = self.image_n_objects[image_index]
            image_object_name_embeddings = np.zeros((n_objects, 300), dtype='float32')
            for object_index in range(n_objects):
                image_object_name_embeddings[object_index] = word_embed(image_nodes[object_index])
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
    def __init__(self, save_dir, n_images, image_n_objects, image_n_ocr, ocr):
        self.save_dir = os.path.join(save_dir, 'targets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.n_images = n_images
        self.image_n_ocr = image_n_ocr
        self.image_n_objects = image_n_objects
        self.ocr = ocr

    def generate(self):
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='target generation'):
            n_ocr = self.image_n_ocr[image_index]
            n_objects = self.image_n_objects[image_index]
            n_nodes = n_objects + n_ocr
            image_target = np.zeros((n_nodes, 2), dtype='float32')
            image_target[:n_objects, 0] = 1
            image_target[n_objects:, 1] = 1
            with open(os.path.join(self.save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_target, f)


class DataSet:
    def __init__(self,
                 tier,
                 save_dir,
                 data_root,
                 word_emb_config):
        self.tier = tier

        # create data save dir
        save_dir = os.path.join(save_dir, 'textvqa_{}'.format(tier))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # read ids map
        ids_map_dir = os.path.join(data_root,
                                   'ids_map',
                                   '{}_ids_map.json'.format(tier))
        with open(ids_map_dir, 'r') as f:
            image_ix_to_id = json.load(f)['image_ix_to_id']
        n_images = len(image_ix_to_id)

        # read node json file
        node_dir = os.path.join(data_root,
                                'nodes',
                                '{}_nodes.json'.format(tier))
        with open(node_dir, 'r') as f:
            nodes = json.load(f)
        n_nodes = len(nodes)
        image_n_nodes = {}  # recode number of node for each image
        for image_index in range(n_images):
            image_id = image_ix_to_id[str(image_index)]
            image_n_nodes[image_index] = len(nodes[image_id])

        # read visual feature h5 file
        visual_feature_dir = os.path.join(data_root,
                                          'object_visual_feature',
                                          '{}_objects.h5'.format(tier))
        visual_feature_h5 = h5py.File(visual_feature_dir, 'r')
        n_objects = len(visual_feature_h5['features'])
        image_n_objects = {}
        for image_index in range(n_images):
            image_n_objects[image_index] = len(delete_zero_padding(visual_feature_h5[image_index]))

        # read ocr
        if tier == 'val':
            ocr_dir = os.path.join(data_root,
                                   'ocr',
                                   '{}_ocr.json'.format('train'))
        else:
            ocr_dir = os.path.join(data_root,
                                   'ocr',
                                   '{}_ocr.json'.format(tier))
        with open(ocr_dir, 'r') as f:
            ocr = json.load(f)
        n_ocr = len(ocr)
        image_n_ocr = {}
        for image_index in range(n_images):
            image_id = image_ix_to_id[str(image_index)]
            image_n_ocr[image_index] = len(ocr[image_id])

        # read adjacent matrix
        adj_matrix_dir = os.path.join(data_root,
                                      'adjacent_matrix',
                                      '{}_edge_rdiou.json'.format(tier))
        with open(adj_matrix_dir, 'r') as f:
            adj_matrix = json.load(f)
        image_adj_dim = {}
        for image_index in range(n_images):
            image_id = image_ix_to_id[str(image_index)]
            image_adj_dim[image_index] = len(adj_matrix[image_id])

        # check input data correctness
        assert n_images == n_nodes
        assert n_images == n_objects
        assert n_images == n_ocr
        for image_index in range(n_images):
            assert image_n_nodes[image_index] == image_adj_dim[image_index]
            assert (image_n_objects[image_index] + image_n_ocr[image_index]) == image_n_nodes[image_index]

        self.node_feature = NodeFeature(save_dir,
                                        image_ix_to_id,
                                        n_images,
                                        nodes,
                                        image_n_objects,
                                        ocr,
                                        visual_feature_dir,
                                        word_emb_config)
        self.adjacent_matrix = AdjMatrix(save_dir,
                                         image_ix_to_id,
                                         adj_matrix_dir)
        self.target = Target(save_dir,
                             n_images,
                             image_n_objects,
                             image_n_ocr,
                             ocr)

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
                           config.data_root,
                           config.word_emb_config)
        data_set.generate()


if __name__ == '__main__':
    main()
