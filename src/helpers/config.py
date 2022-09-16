import json
import os
import torch
import sys
sys.path.append('src/helpers/')
from src.helpers.alphabet import Alphabet
from dotmap import DotMap

class Config():
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as f:
            self.arguments = json.load(f)

        # saving
        self.output_dir = self.arguments.get("output_model_dir") or self.arguments.get("output_dir")
        self.writer_dir = os.path.join(self.output_dir, 'runs')
        self.saving_dir = os.path.join(self.output_dir, "saving")

        self.feature_length = self.arguments.get("feature_length") or None

        # alphabet
        self.alphabet_file = self.arguments.get("alphabet_file") or None  # path to JSON file storing alphabet mapping
        self.alphabet = Alphabet(self.alphabet_file)  # set up vocabulary

        # training
        self.device = self.arguments.get("device") or "cuda"
        if self.device == "cuda":
            if not torch.cuda.is_available():
                print("INFO -- no GPU available, using CPU instead")
                self.device = "cpu"


        # saving
        self.model_weights = self.arguments.get("model_weights") or None

        # data parameters
        self.sample_shape = self.arguments.get("sample_shape") or [40, 180]
        self.max_pred_length = self.arguments.get("max_pred_length") or 7  # max. no of chars that can be predicted
        self.dataset_cache_size = self.arguments.get(
            "dataset_cache_size") or 10000  # number of samples to prefetch sequentially when lazy loading is enabled

        # command line output + log
        self.log_interval = self.arguments.get("log_interval") or 1  # write info all log_interval steps to stdout
        self.print_batch = self.arguments.get("print_batch") or False

        # tensorboard
        self.tensorboard_log = self.arguments.get("tensorboard_log") or False

        # random seeds
        self.training_seed = self.arguments.get("training_seed") or 21  # half the truth
        self.inference_seed = self.arguments.get("inference_seed") or 42  # the whole truth
        assert self.training_seed != self.inference_seed, \
            "Both Training and inference seed are {}. Choose disjunct ones " \
            "to avoid identical degradation operations in data sets".format(self.training_seed)


        # augmentations
        dict = self.arguments.get("aug_params")
        self.aug_params = DotMap(dict) if dict is not None else None

        # knowledge_embedding
        self.knowledge_embedding = self.arguments.get("knowledge_embedding") or False
        self.num_degradation_classes = self.arguments.get("num_degradation_classes") if self.knowledge_embedding else None

        if self.knowledge_embedding:
            print("INFO -- Using knowledge embedding with {} classes".format(self.num_degradation_classes))

    def to_dict(self) -> dict:
        """
        Returns the parameters as a dictionary

        :return:
        """
        new_dict = self.__dict__.copy()
        del new_dict['alphabet']
        del new_dict['arguments']
        return new_dict


class TrainConfig(Config):
    def __init__(self, json_file_path):
        super().__init__(json_file_path)

        # saving
        self.output_dir = self.arguments.get("output_model_dir") or self.arguments.get("output_dir")
        self.logfile = None

        # model parameters
        self.feature_length = self.arguments.get("feature_length") or 44  # input sequence length
        self.projection_dim = self.arguments.get(
            "projection_dim") or None  # linear projection of image slice in encoder (no projection if is None)
        self.dim_feedforward = self.arguments.get(
            "dim_feedforward") or None  # size of feedforward blocks in transformer models' encoder layers
        self.n_encoder_layers = self.arguments.get("n_encoder_layers") or 2  # number of layers in encoder
        self.n_decoder_layers = self.arguments.get("n_decoder_layers") or 2  # number of layers in decoder
        self.dropout = self.arguments.get("dropout") or 0.2  # the dropout value
        self.optimizer = self.arguments.get("optimizer") or "Adam"  # model optimizer
        self.learning_rate = self.arguments.get("learning_rate") or 0.0001  # learning rate for model
        self.teacher_forcing = self.arguments.get(
            "teacher_forcing") or 1  # probability per training batch to use teacher forcing -> always use per default!

        # model parameters: only RNN
        self.hidden_size = self.arguments.get("hidden_size") or None  # hidden size for RNN Seq2Seq models

        # model parameters: only Transformer
        self.n_attn_head = self.arguments.get(
            "n_attn_head") or None  # number of heads in the multihead attention blocks (only used for transformer)

        # data parameters
        self.batch_size = self.arguments.get("batch_size") or 32  # training batch size
        self.eval_batch_size = self.arguments.get("eval_batch_size") or 32  # evaluation batch size
        self.epochs = self.arguments.get("epochs") or 1  # number of epochs the model trains for

        self.train_data_dir = self.arguments.get("train_data_dir")
        self.valid_data_dir = self.arguments.get("valid_data_dir")
        self.items_per_set = self.arguments.get("items_per_set")


class TestConfig(Config):
    def __init__(self, json_file_path):
        super().__init__(json_file_path)

        # saving
        self.output_dir = self.arguments.get("output_dir") or None
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
            print("INFO -- created output_dir {}".format(self.output_dir))

        # data parameters
        self.test_data_dir = self.arguments.get("test_data_dir")
        self.test_batch_size = self.arguments.get("test_batch_size") or 32  # training batch size

        # test param for experiments
        self.start_idx = self.arguments.get("start_idx")
        self.end_idx = self.arguments.get("end_idx")


class BaseNetConfig(Config):
    def __init__(self, json_file_path):
        super().__init__(json_file_path)

        # model type
        self.baseline_model = self.arguments.get("baseline_model") or "EfficientNet_LP"
        print("INFO -- Baseline model type: {}".format(self.baseline_model))

        # data params
        self.batch_size = self.arguments.get("batch_size") or 32  # training batch size
        self.eval_batch_size = self.arguments.get("eval_batch_size") or 32  # evaluation batch size
        self.epochs = self.arguments.get("epochs") or 1  # number of epochs the model trains for
        self.train_data_dir = self.arguments.get("train_data_dir")
        self.valid_data_dir = self.arguments.get("valid_data_dir")
        self.items_per_set = self.arguments.get("items_per_set")

        # model params
        dict = self.arguments.get("effnet_params")
        self.effnet_params = DotMap(dict) if dict is not None else None
        self.restored_weights = self.arguments.get("restored_weigths") or None

        # training params
        self.optimizer = self.arguments.get("optimizer") or "Adam"  # model optimizer
        self.learning_rate = self.arguments.get("learning_rate") or 0.0001  # learning rate for model

