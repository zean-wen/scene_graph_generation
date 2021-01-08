class BaseConfig:
    @ classmethod
    def copy_from_args(cls, args):
        for key, value in vars(args).items():
            if hasattr(cls, key):
                setattr(object, key, value)
        return cls()


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

    @ classmethod
    def copy_from_args(cls, args):
        for key, value in vars(args).items():
            if hasattr(cls, key):
                setattr(object, key, value)
        cls.word_emb_config = WordEmbeddingConfig.copy_from_args(args)
        return cls()
