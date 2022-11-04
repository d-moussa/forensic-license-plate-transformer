import torch
import numpy as np
from src.models.seq2seq_transformer.transformer_helpers import generate_square_subsequent_mask
from src.helpers.csv_helpers import append_data_to_csv


def greedy_decode(model, src, max_len, start_symbol, device, alphabet, knowledge_class):
    src = src.to(device)
    k = knowledge_class.to(device)

    # extract features if cnn_stem given
    if model.cnn_stem is not None:
        # src = src.permute(1, 2, 0)  # batch, height, width
        src = src.permute(1, 2, 0).unsqueeze(
            0)  # CNN needs: batch, channel, height, width. unsqueeze since we only process one sample per decoding step -> artificial batch dim
        src = model.cnn_stem.forward(src.to(device))
        src = src.reshape(src.shape[0], -1, src.shape[3])  # batch, new_height (height x activation_maps), width
        src = src.permute(2, 0, 1)

    memory = model.encode(knowledge_class=k, src=src)
    enc_attn_weights = None
    memory = memory.to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)

        out = model.decode(ys, memory, tgt_mask)
        dec_attn_weights = None
        out = out.transpose(0, 1)
        prob = model.linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # next_token = torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)
        if next_word == alphabet.label_mapping.get('<eos>'):
            break
    return ys, enc_attn_weights, dec_attn_weights


def translate(model, src, alphabet, device, max_pred_length : int, knowledge_class : int):

    model.eval()
    BOS_IDX = alphabet.label_mapping.get('<bos>')

    # reshape input to match (w, C, H) order
    if len(src.shape) == 2:  # german data
        src = src.permute(1, 0).unsqueeze(1)
    else:  # czech data
        src = src.permute(1, 2, 0)

    tgt_tokens, enc_attn_weights, dec_attn_weights = greedy_decode(model=model,
                                                                   src=src,
                                                                   max_len=max_pred_length,
                                                                   start_symbol=BOS_IDX,
                                                                   device=device,
                                                                   alphabet=alphabet,
                                                                   knowledge_class=knowledge_class)
    tgt_tokens = tgt_tokens.flatten()

    tgt_tokens = np.array(tgt_tokens.cpu().numpy(), dtype=np.int)[
                 1:]  # don't return <bos>, model was given to start decoding
    preds_string = "".join((list(
        map(lambda x: alphabet.reverse_label_mapping.get(tgt_tokens[x]),
            range(tgt_tokens.size)))))

    return preds_string, enc_attn_weights, dec_attn_weights


def predict_batch(src, tgt, model, alphabet, device, max_pred_length:int, knowledge_class:int):
    """
    inferences model on one batch of data and returns predictions and targets
    Abbreviations: B: batch Size, H: height, W: width, C: channels, L: Sequence Length
    :param src: input data to predict on, Shape: (B, H, W, C)
    :param tgt: labels for input data, Shape: (B, L)
    :param model: model used for inferencing
    :param alphabet: alphabet for model
    :param device: device to run on ("cpu", "cuda:0", ...)
    :param knowledge_class: additional information fed to model (int encoding class)
    :return: prediction and targets for whole batch, attention weights per batch if provided, else []
    """
    model.eval()
    tgt = np.array(tgt.numpy(), dtype=np.int)
    predictions = []
    targets = []
    enc_attn_weights = []
    dec_attn_weights = [[], []]

    for i in range(src.shape[0]):
        preds, enc_attn_w, dec_attn_w = translate(model=model,
                                                  src=src[i],
                                                  alphabet=alphabet,
                                                  device=device,
                                                  max_pred_length=max_pred_length,
                                                  knowledge_class=knowledge_class[i])
        predictions.append(preds)
        tgt_item = tgt[i]
        targets.append("".join(
            list(map(lambda x: alphabet.reverse_label_mapping.get(tgt_item[x]), range(tgt_item.size)))))

    return predictions, targets, enc_attn_weights, dec_attn_weights


def predict_to_csv(prediction_file, dataloader, h5_keys, model, alphabet, batch_size, device, max_pred_length):
    """
    predicts on test data with given model and writes outputs to file
    :param prediction_file: file to write to, predictions will be appended
    :param dataloader: torch dataloader containing test data
    :param h5_keys: h5 info to be extracted and written to prediciton file
    :param model: neural network model to test
    :param alphabet: alphabet for model
    :param batch_size: test batch size
    :param device: device to run on ("cpu", "cuda:0", ...)
    :return:
    """
    model.eval()
    test_data_iter = iter(dataloader)
    for batch_idx, (k, src, tgt) in enumerate(test_data_iter):
        pred_batch = {}

        predictions, targets, _, _ = predict_batch(src=src,
                                                   tgt=tgt,
                                                   model=model,
                                                   alphabet=alphabet,
                                                   device=device,
                                                   max_pred_length=max_pred_length,
                                                   knowledge_class=k)

        pred_batch["predictions"] = predictions
        pred_batch["license_numbers"] = targets
        # get additional info about samples from test data set
        start_idx = batch_idx * batch_size

        if h5_keys is not None:
            for k in h5_keys:
                v = dataloader.dataset.data_file[k][start_idx:start_idx + len(predictions)]
                pred_batch[k] = v
        append_data_to_csv(prediction_file, pred_batch)
