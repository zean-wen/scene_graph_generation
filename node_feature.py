import os
import argparse

from tqdm import tqdm
import h5py
import json
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='train_val_test', help='train, val, test')
    parser.add_argument('--save_folder', type=str, help='path to save result')
    parser.add_argument('--ids_map_folder', type=str, help='image index map file')
    parser.add_argument('--scene_graph_folder', type=str, help='file path save the scene graph')
    parser.add_argument('--visual_feature_folder', type=str, help='path where the object feature saved')
    parser.add_argument('--ocr_folder', type=str, help='path where ocr saved')
    parser.add_argument('--glove_dictionary_file', type=str, help='dictionary file')
    parser.add_argument('--glove_word_matrix_file', type=str, help='word embedding matrix file')
    parser.add_argument('--fasttext_dictionary_file', type=str, help='dictionary file')
    parser.add_argument('--fasttext_word_matrix_file', type=str, help='word embedding matrix file')

    return parser.parse_args()


class WordEmbeddingConfig:
    glove_dictionary_file: str
    glove_word_matrix_file: str
    fasttext_dictionary_file: str
    fasttext_word_matrix_file: str


class Config:
    tiers: str
    save_folder: str
    ids_map_folder: str
    scene_graph_folder: str
    ocr_folder: str
    visual_feature_folder: str
    word_embed_config: WordEmbeddingConfig = WordEmbeddingConfig()

    def parse_from_args(self, args):
        self.tiers = args.tiers
        self.save_folder = args.save_folder
        self.ids_map_folder = args.ids_map_folder
        self.scene_graph_folder = args.scene_graph_folder
        self.ocr_folder = args.ocr_folder
        self.visual_feature_folder = args.visual_feature_folder
        self.word_embed_config.glove_dictionary_file = args.glove_dictionary_file
        self.word_embed_config.glove_word_matrix_file = args.glove_word_matrix_file
        self.word_embed_config.fasttext_dictionary_file = args.fasttext_dictionary_file
        self.word_embed_config.fasttext_word_matrix_file = args.fasttext_word_matrix_file


class WordEmbedding:
    def __init__(self, embedding_method, config):
        if embedding_method.lower() == 'glove':
            dictionary_file = config.glove_dictionary_file
            word_matrix_file = config.glove_word_matrix_file
        elif embedding_method.lower() == 'fasttext':
            dictionary_file = config.fasttext_dictionary_file
            word_matrix_file = config.fasttext_word_matrix_file
        else:
            raise ValueError('{} embedding method is allowed'.format(embedding_method))
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


class NodeFeature:
    def __init__(self,
                 h5_file_dir,
                 ids_map_json,
                 scene_graph_json,
                 visual_feature_h5,
                 ocr_json,
                 word_emb_config):

        self.h5_file = h5py.File(h5_file_dir, 'w')
        with open(ids_map_json, 'r') as f:
            self.image_ix_to_id = json.load(f)['image_ix_to_id']
        self.scene_graph_json = scene_graph_json
        self.visual_feature_h5 = visual_feature_h5
        self.ocr_json = ocr_json
        self.n_images = len(self.image_ix_to_id)
        self.word_emb_config = word_emb_config

    def build(self):
        self.object_name_embedding_generation()
        self.object_visual_feature_generation()
        self.ocr_feature_generation()
        self.h5_file.close()

    def object_name_embedding_generation(self):
        self.h5_file.create_dataset("object_name_embedding", (self.n_images, 36, 300), dtype='float32')
        word_embed = WordEmbedding('glove', self.word_emb_config)
        with open(self.scene_graph_json, 'r') as f:
            scene_graphs = json.load(f)
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Object name embedding generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_scene_graph = scene_graphs[image_id]
            image_object_name_embedding = np.zeros((36, 300), dtype='float32')
            for object_index, _object in enumerate(image_scene_graph['objects'].values()):
                image_object_name_embedding[object_index] = word_embed(_object['name'])
            self.h5_file["object_name_embedding"][image_index] = image_object_name_embedding
        del scene_graphs

    def object_visual_feature_generation(self):
        self.h5_file.create_dataset("object_visual_features", (self.n_images, 36, 2048), dtype='float32')
        object_visual_features = h5py.File(self.visual_feature_h5, 'r')
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Object visual feature generation'):
            self.h5_file['object_visual_features'][image_index] = object_visual_features['features'][image_index]

        object_visual_features.close()

    def ocr_feature_generation(self):
        with open(self.ocr_json, 'r') as f:
            ocr = json.load(f)
        max_len = self.find_image_ocr_max_len(ocr)

        word_embed = WordEmbedding('fasttext', self.word_emb_config)

        self.h5_file.create_dataset("ocr_token_embeddings", (self.n_images, max_len, 300), dtype='float32')
        self.h5_file.create_dataset("ocr_bounding_boxes", (self.n_images, max_len, 8), dtype='float32')
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Ocr feature generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_ocr = ocr[image_id]
            image_ocr_token_embedding = np.zeros((max_len, 300), dtype='float32')
            image_ocr_bounding_boxes = np.zeros((max_len, 8), dtype='float32')
            for ocr_index, (ocr_token, bbox) in enumerate(image_ocr.items()):
                image_ocr_token_embedding[ocr_index] = word_embed(ocr_token)
                image_ocr_bounding_boxes[ocr_index] = np.array(bbox).flatten()
            self.h5_file['ocr_token_embeddings'][image_index] = image_ocr_token_embedding
            self.h5_file['ocr_bounding_boxes'][image_index] = image_ocr_bounding_boxes

    @staticmethod
    def find_image_ocr_max_len(ocr):
        max_len = 0
        for image_ocr in ocr.values():
            if max_len < len(image_ocr):
                max_len = len(image_ocr)
        return max_len


def main():
    args = get_args()
    config = Config()
    config.parse_from_args(args)

    tiers = config.tiers.split('_')
    for tier in tiers:
        print('#### Generating node feature for {} images ####'.format(tier))
        h5_file_dir = os.path.join(config.save_folder, '{}_node_features.h5'.format(tier))
        ids_map_json = os.path.join(config.ids_map_folder, '{}_ids_map.json'.format(tier))
        scene_graph_json = os.path.join(config.scene_graph_folder, '{}_sg.json'.format(tier))
        visual_feature_h5 = os.path.join(config.visual_feature_folder, '{}_objects.h5'.format(tier))
        if tier == 'val':
            ocr_json = os.path.join(config.ocr_folder, '{}_ocr.json'.format('train'))
        else:
            ocr_json = os.path.join(config.ocr_folder, '{}_ocr.json'.format(tier))

        node_feature = NodeFeature(h5_file_dir,
                                   ids_map_json,
                                   scene_graph_json,
                                   visual_feature_h5,
                                   ocr_json,
                                   config.word_embed_config)
        node_feature.build()


if __name__ == '__main__':
    main()
