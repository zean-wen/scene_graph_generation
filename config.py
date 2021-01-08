class BaseConfig:
    def copy_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(object, key, value)


class WordEmbeddingConfig(BaseConfig):
    glove_dictionary_file: str
    glove_word_matrix_file: str
    fasttext_dictionary_file: str
    fasttext_word_matrix_file: str


class Config(BaseConfig):
    tiers: str
    save_dir: str
    ids_map_dir: str
    data_root: str
    scene_graph_folder: str
    ocr_folder: str
    visual_feature_folder: str
    word_emb_config: WordEmbeddingConfig = WordEmbeddingConfig()

    def copy_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(object, key, value)
        self.word_emb_config.copy_from_args(args)
