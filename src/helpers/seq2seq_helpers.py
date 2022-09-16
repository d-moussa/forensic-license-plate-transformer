import json
import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.helpers.config import TrainConfig, BaseNetConfig


def tensorboard_logger(writer, scalar: str, value, step: int):
    writer.add_scalar(scalar, value, step)


def insert_tokens(src, tgt, alphabet):
    """
    inserts special tokens eos, bos and pad into input and label sequence
    :param src: source sequence
    :param tgt: target sequence
    :param alphabet: Alphabet object
    :return:
    """
    img_batch, label_batch = [], []

    BOS_IDX = alphabet.label_mapping.get('<bos>')
    EOS_IDX = alphabet.label_mapping.get('<eos>')
    PAD_IDX = alphabet.label_mapping.get('<pad>')

    assert (not None in [BOS_IDX, EOS_IDX, PAD_IDX]), "Alphabet File is missing BOS, EOS or PAD Token"

    for b_idx in range(len(src)):
        label_batch.append(torch.cat([torch.tensor([BOS_IDX]), tgt[b_idx], torch.tensor([EOS_IDX])], dim=0))

    img_batch = pad_sequence(src, padding_value=PAD_IDX, batch_first=True)
    label_batch = pad_sequence(label_batch, padding_value=PAD_IDX, batch_first=True)
    return img_batch, label_batch


def print_batch(output, targets, parameters, batch_idx):

    print("INFO -- start of batch {}".format(batch_idx))
    for pred, target in zip(output, targets):
        preds_numerical = torch.argmax(pred, dim=1).cpu().numpy()
        target = target.cpu().numpy()
        preds_string = "".join((list(
            map(lambda x: parameters.alphabet.reverse_label_mapping.get(preds_numerical[x]),
                range(preds_numerical.size)))))
        target_string = "".join(
            list(map(lambda x: parameters.alphabet.reverse_label_mapping.get(target[x]), range(target.size))))

        print("label: {}, prediction: {}".format(target_string, preds_string))
    print("INFO -- end of batch")


def prolog(arg: str, baseline: bool = False):
    """
    make fundamental initializations
    :param arg:
    :return:
    """
    # load training parameters
    if baseline:
        parameters = BaseNetConfig(arg)
    else:
        parameters = TrainConfig(arg)

    # set random seeds
    random.seed(parameters.training_seed)
    np.random.seed(parameters.training_seed)
    torch.manual_seed(parameters.training_seed)

    # create output directory
    export_config_filename = os.path.join(parameters.output_dir, 'config.json')

    if not parameters.model_weights:
        # check if output folder already exists
        assert not os.path.isdir(parameters.output_dir), \
            '{} already exists, specify another directory via config file.'.format(parameters.output_dir)
        os.makedirs(parameters.output_dir)
        os.makedirs(parameters.saving_dir)

    # export config file in model output dir
    with open(export_config_filename, 'w') as file:
        json.dump(parameters.to_dict(), file)

    # make logfile path
    parameters.logfile = os.path.join(parameters.output_dir, 'log.txt')

    return parameters
