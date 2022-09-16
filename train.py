import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_loading.augmentations import Normalize, Resolution, JpegCompression, PerImageStandardizationTransform, \
    ScaleToNetInputSize
from src.data_loading.synthetic_data_set_gen import SyntheticDataGenerator
from src.helpers.seq2seq_helpers import prolog
from src.models.seq2seq_transformer.transformer_helpers import construct_model
from src.models.seq2seq_transformer.transformer_training import train_epoch, evaluate

# some basic initializations
parameters = prolog(str(sys.argv[1]))

# create writer for summarization on tensorboard
if parameters.tensorboard_log:
    os.makedirs(parameters.writer_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=parameters.writer_dir)
else:
    writer = None

# define augmentations
if parameters.aug_params.augment:
    transforms = [Normalize(),
                  Resolution(parameters.aug_params, parameters.sample_shape),
                  JpegCompression(parameters.aug_params, linear_map_factor=100//parameters.num_degradation_classes),
                  ScaleToNetInputSize(parameters.sample_shape, parameters.aug_params),
                  PerImageStandardizationTransform()]
else:
    transforms = [PerImageStandardizationTransform()]

print("INFO -- applying dataset transforms {}".format(transforms))

# create datasets
train_path = parameters.train_data_dir
eval_path = parameters.valid_data_dir
train_dataset = SyntheticDataGenerator(train_path, parameters, mode='train', transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=False)

# init model
model = construct_model(model_params=parameters,
                        model_weights=parameters.model_weights,
                        alphabet_size=parameters.alphabet.size,
                        device=parameters.device)
model = model.to(parameters.device)
print("INFO -- training_seed: {}, inference_seed: {} ".format(parameters.training_seed, parameters.inference_seed))
print("INFO -- device: {}".format(parameters.device))
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# set up loss, optimizer, learning rate (lr) and scheduler
criterion = torch.nn.CrossEntropyLoss(ignore_index=parameters.alphabet.label_mapping.get('<pad>'))
optimizer = getattr(optim, parameters.optimizer)(model.parameters(), lr=parameters.learning_rate)
print("INFO -- Initialized optimizer {} with lr: {}".format(parameters.optimizer, parameters.learning_rate))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
print("INFO -- LR Optimizer: ReduceLROnPlateau")

# Adjust the learning rate after each epoch.
for epoch in range(parameters.epochs):
    train_dataset = SyntheticDataGenerator(train_path, parameters, mode='train', transform=transforms)
    eval_dataset = SyntheticDataGenerator(eval_path, parameters, mode='valid', transform=transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=parameters.eval_batch_size, shuffle=True)

    epoch_start_time = time.time()

    print("INFO -- Start of training step")
    # training step
    train_epoch(parameters=parameters, model=model, train_dataloader=train_dataloader, optimizer=optimizer,
               criterion=criterion, epoch=epoch, writer=writer)

    print("INFO -- End of training step")
    print("| Epoch {} completed in {} |".format(epoch, time.strftime('%Y-%m-%d %H:%M:%S',
                                                                     time.localtime(time.time() - epoch_start_time))[
                                                       -8:]))
    print("INFO -- Start of evaluation step")

    # evaluation step
    val_loss, val_acc = evaluate(
        eval_model=model,
        data_source=eval_dataloader,
        no_samples=len(eval_dataset),
        criterion=criterion,
        parameters=parameters)
    print('-' * 89)
    string1 = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' \
              'valid accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, val_acc)
    print(string1)
    print('-' * 89)
    f = open(parameters.logfile, 'a')
    f.write('-' * 89 + '\n' + string1 + '\n' + '-' * 89 + '\n')
    f.close()
    print("INFO -- End of evaluation step")

    scheduler.step(val_loss)
    if parameters.tensorboard_log:
        writer.add_scalar("lr (epoch wise)", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    torch.save(model.state_dict(), os.path.join(parameters.saving_dir, "model_weights-{}".format(epoch) + ".pth"))

if parameters.tensorboard_log:
    writer.close()
print("INFO -- Training Finished")
timestamp = str(int(time.time()))

torch.save(model.state_dict(), os.path.join(parameters.saving_dir, "model_weights" + ".pth"))
saving_path = os.path.join(parameters.saving_dir, "model" + ".pth")
torch.save(model, saving_path)
print("INFO -- model saved to {}".format(saving_path))
