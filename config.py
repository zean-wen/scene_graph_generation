class BaseConfig:
    @ classmethod
    def copy_from_args(cls, args):
        for key, value in vars(args).items():
            if hasattr(cls, key):
                setattr(object, key, value)
        return cls()


class WordEmbeddingConfig(BaseConfig):
    glove_dictionary_file: str = None
    glove_word_matrix_file: str = None
    fasttext_dictionary_file: str = None
    fasttext_word_matrix_file: str = None


class Config(BaseConfig):
    tiers: str = None
    save_dir: str = None
    ids_map_dir: str = None
    data_root: str = None
    scene_graph_folder: str = None
    ocr_folder: str = None
    visual_feature_folder: str = None
    word_emb_config: WordEmbeddingConfig = WordEmbeddingConfig()

    @ classmethod
    def copy_from_args(cls, args):
        for key, value in vars(args).items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        cls.word_emb_config = WordEmbeddingConfig.copy_from_args(args)
        return cls()
