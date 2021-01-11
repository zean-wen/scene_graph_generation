import os


class WordEmbeddingConfig:
    def __init__(self, data_root):
        self.glove_dictionary_file = os.path.join(data_root, 'word_embedding', 'glove_dictionary.json')
        self.glove_word_matrix_file = os.path.join(data_root, 'word_embedding', 'glove6b_init_300d.npy')
        self.fasttext_dictionary_file = os.path.join(data_root, 'word_embedding', 'fasttext_init_300d.npy')
        self.fasttext_word_matrix_file = os.path.join(data_root, 'word_embedding', 'fasttext_dictionary.json')


class Config:
    def __init__(self, args):
        self.tiers = args.tiers
        self.data_root = args.data_root
        self.save_dir = args.data_root
        self.word_emb_config: WordEmbeddingConfig = WordEmbeddingConfig(self.data_root)

