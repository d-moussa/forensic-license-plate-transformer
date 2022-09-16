import torch
from src.helpers.seq2seq_helpers import print_batch, insert_tokens
from src.models.seq2seq_transformer.transformer_helpers import create_masks, generate_square_subsequent_mask


def train_batch(src, tgt, parameters, model, optimizer, criterion, k):
    model.train()

    # extract features if cnn stem given
    if model.cnn_stem is not None:
        src = src.permute(0, 3, 1, 2)  # CNN needs: batch, channel, height, width.
        src = model.cnn_stem.forward(src.to(parameters.device))
        src = src.reshape(src.shape[0], -1, src.shape[3])  # batch, new_height (height x activation_maps), width
        src = src.unsqueeze(3)

    # insert special tokens into sequence (eos, bos, pad)
    src = src.squeeze().permute(0, 2, 1)
    src, tgt = insert_tokens(src, tgt, parameters.alphabet)

    # transform data to match transformer requirements (Seq length, batch size, feature size)
    src = src.permute(1, 0, 2).to(parameters.device)
    tgt = tgt.permute(1, 0).to(parameters.device)

    src = src.to(parameters.device)
    tgt = tgt.to(parameters.device)
    k = k.to(parameters.device)

    # prepare tgt input and padding masks
    tgt_input = tgt[:-1, :]
    src_padding_mask, tgt_padding_mask, tgt_mask = create_masks(src, tgt_input, parameters, batch_dim=src.shape[1])

    # run model
    logits, attn_weights = model(src=src,
                                 trg=tgt_input,
                                 tgt_mask=tgt_mask,
                                 src_padding_mask=src_padding_mask,
                                 tgt_padding_mask=tgt_padding_mask,
                                 memory_key_padding_mask=src_padding_mask,
                                 knowledge_class=k)

    tgt_out = tgt[1:, :]

    # reshape vectors to move batch dimension to front
    output = logits.permute(1, 0, 2)
    targets = tgt_out.permute(1, 0)

    # compute batch loss based upon predictions
    batch_loss = 0.0
    correct_preds = 0
    for pred, target in zip(output, targets):
        batch_loss += criterion(pred, target)  # included <eos> here!
        if torch.equal(torch.argmax(pred, dim=1), target):
            correct_preds += 1
    batch_loss /= parameters.batch_size
    batch_accuracy = (correct_preds / parameters.batch_size)
    batch_loss.backward()
    optimizer.step()

    return output, targets, batch_loss.item(), batch_accuracy


def train_epoch(parameters, model, train_dataloader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0.
    total_acc = 0.
    train_data_iter = iter(train_dataloader)
    for batch_idx, (k, src, tgt) in enumerate(train_data_iter):

        # step counter for logging
        curr_step = epoch * len(train_dataloader) + batch_idx

        optimizer.zero_grad()
        # train on batch
        output, targets, batch_loss, batch_acc = train_batch(src=src,
                                                             tgt=tgt,
                                                             parameters=parameters,
                                                             model=model,
                                                             optimizer=optimizer,
                                                             criterion=criterion,
                                                             k=k)
        total_loss += batch_loss
        total_acc += batch_acc

        # log training loss
        if batch_idx != 0 and batch_idx % parameters.log_interval == 0:
            string_1 = '| epoch: {:3d} | {:5d}/{:5d} batches | lr: {:.1E}  | training batch loss:' \
                       ' {:5.2f} | training batch acc: {:5.2f}'.format(epoch, batch_idx, len(train_dataloader),
                                                                       optimizer.param_groups[0]['lr'],
                                                                       batch_loss,
                                                                       batch_acc * 100)
            print(string_1)
            f = open(parameters.logfile, 'a')
            f.write(string_1 + '\n')
            f.close()

            # Print a batch of predictions
            if parameters.print_batch:
                print("INFO -- Print training batch")
                print_batch(output, targets, parameters, batch_idx)

        if parameters.tensorboard_log:
            # log loss and accuracy on tensorboard
            writer.add_scalar('Loss/train', batch_loss, curr_step)
            writer.add_scalar('Accuracy/train', batch_acc, curr_step)

    return (total_loss / (batch_idx + 1)), (total_acc / (batch_idx + 1)) * 100


def evaluate_batch(model, src, tgt, criterion, parameters, k):
    model.eval()
    batch_loss = 0.
    correct_preds = 0.
    output = torch.zeros((parameters.eval_batch_size, parameters.max_pred_length + 1, parameters.alphabet.size))
    BOS_IDX = parameters.alphabet.label_mapping.get('<bos>')

    # extract features if cnn stem given
    if model.cnn_stem is not None:
        src = src.permute(0, 3, 1, 2)  # CNN needs: batch, channel, height, width.
        src = model.cnn_stem.forward(src.to(parameters.device))
        src = src.reshape(src.shape[0], -1, src.shape[3])  # batch, new_height (height x activation_maps), width
        src = src.unsqueeze(3)

    # insert special tokens into sequence (eos, bos, pad)
    src = src.squeeze().permute(0, 2, 1)
    src, tgt = insert_tokens(src, tgt, parameters.alphabet)
    k = k.to(parameters.device)

    targets = tgt[:, 1:]

    for s in range(src.shape[0]):
        sample = src[s].unsqueeze(dim=1).to(parameters.device)

        # inference model
        memory = model.encode(sample, k[s])
        memory = memory.to(parameters.device)
        ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(parameters.device)

        # we do not break here if <eos> is reached -> change for variable license plate lengths
        for i in range(parameters.max_pred_length + 1):
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), parameters.device)
                        .type(torch.bool)).to(parameters.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            logits = model.linear(out[:, -1])
            output[s, i, :] = (logits.squeeze())

            # append character prediction to sequence
            _, next_word = torch.max(logits, dim=1)
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word.item()).to(parameters.device)], dim=0)

    # update loss and no. correct predictions
    for pred, target in zip(output, targets):
        batch_loss += criterion(pred, target)
        if torch.equal(torch.argmax(pred, dim=1), target):
            correct_preds += 1

    return batch_loss, correct_preds, output, targets


def evaluate(eval_model, data_source, no_samples, criterion, parameters):
    eval_model.eval()
    total_loss = 0.
    first = True
    correct_preds = 0
    with torch.no_grad():
        # iterate over all batches in eval data set
        for k, data, labels in data_source:
            cum_loss, batch_correct_preds, output, targets = evaluate_batch(model=eval_model,
                                                                            src=data,
                                                                            tgt=labels,
                                                                            criterion=criterion,
                                                                            parameters=parameters,
                                                                            k=k)
            correct_preds += batch_correct_preds
            total_loss += cum_loss
            if parameters.print_batch and first:
                print("INFO -- Printing first validation batch")
                print_batch(output, targets, parameters, batch_idx=0)
                first = False

        total_acc = (correct_preds / no_samples) * 100
        total_loss /= no_samples

    return total_loss, total_acc
