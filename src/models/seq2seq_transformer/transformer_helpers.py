import sys
import numpy as np
import torch
sys.path.append('src/models/seq2seq_transformer/')
from src.models.seq2seq_transformer.seq2seq_model import Seq2SeqTransformer


def translate_to_codes(caption, parameters):
    """
    Gets label codes from string captions. If len(caption) < max_chars, pad with <pad>
    :param caption: string to translate to numerical representation
    :param parameters: train config params
    :return: numerical caption
    """

    diff = parameters.max_pred_length - len(caption)
    cap_code = np.array(list(map(lambda x: parameters.alphabet.label_mapping.get(x), caption)))
    cap_code = np.pad(cap_code, (0, diff), 'constant',
                      constant_values=(parameters.alphabet.label_mapping.get('<pad>')))
    return cap_code


# generate masks to prevent attending to all time steps
# should be zero mask for src, but square mask for target in our case
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


# mask generation for transformer input
def create_masks(src, tgt, parameters, batch_dim):

    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, parameters.device)

    src_padding_mask = (src[:, :, 0] == parameters.alphabet.label_mapping.get('<pad>')).permute(1, 0).to(
        parameters.device)

    tgt_padding_mask = (tgt == parameters.alphabet.label_mapping.get('<pad>')).permute(1, 0).to(parameters.device)

    return src_padding_mask, tgt_padding_mask, tgt_mask


def construct_model(model_params, model_weights, alphabet_size, device):
    """given the model parameters, device and model_weights constructs and returns the model

    :param model_params: Config Object containing all model parameters
    :param model_weights: model weights that shall be loaded, don't load any if None
    :param alphabet_size: size of the target vocabulary
    :param device: device to move model to
    """

    print("INFO -- Using Torch Transformer Layers")
    model = Seq2SeqTransformer(tgt_vocab_size=alphabet_size, feature_length=model_params.feature_length,
                               projection_dim=model_params.projection_dim, nhead=model_params.n_attn_head,
                               dim_feedforward=model_params.dim_feedforward,
                               num_encoder_layers=model_params.n_encoder_layers,
                               num_decoder_layers=model_params.n_decoder_layers, dropout=model_params.dropout,
                               cnn_stem=None, knowledge_classes=model_params.num_degradation_classes).to(device)

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, map_location=device))
        print("INFO -- restoring model with weights {}".format(model_weights))

    return model
