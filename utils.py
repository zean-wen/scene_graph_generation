import json
import numpy as np


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


def delete_zero_padding(image_object_visual_features):
    image_object_visual_features = image_object_visual_features[~np.all(
        image_object_visual_features == 0,
        axis=1)]
    return image_object_visual_features
