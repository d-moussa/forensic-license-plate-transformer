import torch
import sys
sys.path.append('src/models/seq2seq_transformer')
sys.path.append('src/models/seq2seq_transformer')
sys.path.append('src/helpers/')
sys.path.append('src/data_loading/')
from src.data_loading.augmentations import Normalize, JpegCompression, Resolution, PerImageStandardizationTransform, ScaleToNetInputSize
import os
import time
import random
import numpy as np
from src.helpers.config import TestConfig,TrainConfig
from src.data_loading.synthetic_data_set_gen import SyntheticDataGenerator
from src.models.seq2seq_transformer.transformer_inference import predict_to_csv
from src.models.seq2seq_transformer.transformer_helpers import construct_model
from torch.utils.data import DataLoader
from src.helpers.csv_helpers import create_csv_file


# load test parameters
parameters = TestConfig(str(sys.argv[1]))

# load model parameters
model_dir = parameters.model_weights.split("saving")[0]
model_config_file = os.path.join(model_dir, "config.json")
model_params = TrainConfig(model_config_file)
print("INFO -- loading model with parameters from {}".format(model_config_file))


# set random seeds
random.seed(parameters.inference_seed)
np.random.seed(parameters.inference_seed)
torch.manual_seed(parameters.inference_seed)

# load model and set to inference mode
model = construct_model(model_params=model_params,
                model_weights=parameters.model_weights,
                alphabet_size=parameters.alphabet.size,
                device=parameters.device)


model.to(parameters.device)
model.eval()
print("INFO -- training_seed: {}, inference_seed: {} ".format(parameters.training_seed, parameters.inference_seed))
print("INFO -- device: {}".format(parameters.device))
print("INFO -- model {} loaded".format(parameters.model_weights))

print("INFO -- Start testing on dir: {}".format(parameters.test_data_dir))

os.makedirs(parameters.output_dir, exist_ok=True)


# create dataset
if parameters.aug_params.augment:

    jpeg_transform = JpegCompression(parameters.aug_params, linear_map_factor=100//model_params.num_degradation_classes)

    transforms = [Normalize(), Resolution(parameters.aug_params, parameters.sample_shape),
                  jpeg_transform,  ScaleToNetInputSize(parameters.sample_shape, parameters.aug_params),
                  PerImageStandardizationTransform()]

else:
    transforms = [PerImageStandardizationTransform()]

test_dataset = SyntheticDataGenerator(parameters.test_data_dir, model_params, mode='test', transform=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False)

timestamp = str(int(time.time()))
prediction_file = os.path.join(parameters.output_dir, "predictions-{}.csv".format(timestamp))

# specify columns of csv file
columns = (['predictions', 'license_numbers'])


create_csv_file(prediction_file, columns)

print("INFO -- start predicting to: {}".format(prediction_file))
predict_to_csv(prediction_file=prediction_file,
               dataloader= test_dataloader,
               h5_keys= None,
               model= model,
               alphabet= parameters.alphabet,
               batch_size= parameters.test_batch_size,
               device= parameters.device,
               max_pred_length=parameters.max_pred_length)
print("INFO -- Finished. Wrote output to {}".format(prediction_file))
