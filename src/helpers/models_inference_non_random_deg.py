import sys
import torch
sys.path.append('models')
sys.path.append('src/utils/')
sys.path.append('src/data_loading/')
from src.data_loading.augmentations import Normalize, CustomJpegCompression, \
    PerImageStandardizationTransform, ScaleToNetInputSize
from src.data_loading.augmentations import Resolution
import os
import time
import random
import numpy as np
from src.helpers.config import TestConfig, BaseNetConfig, TrainConfig
from src.data_loading.synthetic_data_set_gen import SyntheticDataGenerator
from src.models.baseline_models.baseline_models_training import predict_to_csv as baseline_predict_to_csv
from src.models.seq2seq_transformer.transformer_inference import predict_to_csv
from src.models.seq2seq_transformer.transformer_helpers import construct_model
from src.models.baseline_models.cnn_models import EfficientNet_LP, LorchCNN, SpanhelCNN, SpanhelCNNSmall

from torch.utils.data import DataLoader
from src.helpers.csv_helpers import create_csv_file

# general Globals
DIR_NAME = 'res_{}_jpeg_{}'
CNN_BASED_ARCHITECTURE = False  # Model Type to Use
NUM_QUALITY_CLASSES = 1  # class injection to perform (from [0, Num_quality_classes]), 1 means no injection

# Degradation Globals
H264_PARAMS = [51]#reversed(list(np.arange(1, 52, 2)))
jpeg_4_abtastung = np.concatenate((np.arange(1,100,4), [100]))
jpeg_missing_to_4 = [9, 13, 17,29, 33, 37,49, 53, 57, 69, 73, 77,89, 93, 97]

JPEG_PARAMS = [1, 5, 10, 11, 15, 20, 21, 25, 30, 31, 35, 40, 41, 45, 50, 51, 55, 60, 61, 65, 70, 71, 75, 80, 81, 85, 90,
               91, 95, 100]
JPEG_PARAMS = jpeg_4_abtastung
RES_PARAMS = np.arange(20, 181, 1)

# Inf4former Globals
BIN_MAP = None  # jpeg_qf_bin_maps.BIN_MAP_21
LINEAR_MAP_FACTOR = None

# load test parameters
parameters = TestConfig(str(sys.argv[1]))
PREDICTION_BASE_DIR = parameters.output_dir
START_IDX = parameters.start_idx
END_IDX = parameters.end_idx
assert START_IDX is not None and END_IDX is not None

# manually set aug_params to ensure framework compatibility
parameters.aug_params.H264.probability = 1
parameters.aug_params.Jpeg.probability = 1
parameters.aug_params.Resolution.probability = 1
parameters.aug_params.Resolution.keep_aspect_ratio = True
parameters.aug_params.Resolution.interpolation = ["INTER_LINEAR"]

# load model parameters
model_dir = parameters.model_weights.split("saving")[0]
model_config_file = os.path.join(model_dir, "config.json")

if CNN_BASED_ARCHITECTURE:
    model_params = BaseNetConfig(model_config_file)
else:
    model_params = TrainConfig(model_config_file)
print("INFO -- loading model with parameters from {}".format(model_config_file))

# set random seeds
random.seed(parameters.inference_seed)
np.random.seed(parameters.inference_seed)
torch.manual_seed(parameters.inference_seed)

# init model
assert parameters.model_weights is not None
if parameters.baseline_model is not None:
    print("INFO -- Baseline model type: {}".format(parameters.baseline_model))
    if parameters.baseline_model == "EfficientNet_LP":
        model = EfficientNet_LP(channels=model_params.effnet_params.channels,
                                ff_dim=model_params.effnet_params.ff_dim,
                                pretrained=False,
                                alphabet_size=parameters.alphabet.size,
                                device=parameters.device,
                                version='0')
    elif parameters.baseline_model == "LorchCNN":
        model = LorchCNN(alphabet_size=parameters.alphabet.size,
                         device=parameters.device)
    elif parameters.baseline_model == "SpanhelCNN":
        model = SpanhelCNN(alphabet_size=parameters.alphabet.size,
                           device=parameters.device)
    elif parameters.baseline_model == "SpanhelCNNSmall":
        model = SpanhelCNNSmall(alphabet_size=parameters.alphabet.size,
                                device=parameters.device)
    else:
        print("ERROR -- baseline_model must be one of [EfficientNet_LP, Lorch_CNN, SpanhelCNN, SpanhelCNNSmall]")
        exit(1)

    model.load_state_dict(torch.load(parameters.model_weights, map_location=parameters.device))
    model.to(parameters.device)
else:
    model = construct_model(model_params=model_params,
                            model_weights=parameters.model_weights,
                            alphabet_size=parameters.alphabet.size,
                            device=parameters.device)
model.eval()
print("INFO -- model {} loaded".format(parameters.model_weights))

print("INFO -- Start testing on dir: {}".format(parameters.test_data_dir))
print("INFO -- Covering JPEG quality factors {}".format(JPEG_PARAMS[START_IDX:END_IDX + 1]))


print("INFO -- RES factors {}".format(RES_PARAMS))

for i in range(START_IDX, END_IDX + 1):

    for r in range(len(RES_PARAMS)):
        # create output prediction file in a subdir named like the test dir (to distinguish prediction output later)

        base_pred_output_dir = os.path.join(PREDICTION_BASE_DIR, DIR_NAME.format(RES_PARAMS[r], JPEG_PARAMS[i]))

        os.makedirs(base_pred_output_dir, exist_ok=True)

        for j in range(0, NUM_QUALITY_CLASSES):
            if NUM_QUALITY_CLASSES == 1:
                pred_output_dir = base_pred_output_dir
            else:
                # create output prediction file in a subdir named like the test dir (to distinguish prediction output later)
                pred_output_dir = os.path.join(base_pred_output_dir, 'inf_class_{}'.format(j))
            os.makedirs(pred_output_dir, exist_ok=True)

            # create daataset
            if parameters.aug_params.augment:
                # manually set aug parameters to ensure framework compatibility
                parameters.aug_params.Resolution.min_res = RES_PARAMS[r]
                parameters.aug_params.Resolution.max_res = RES_PARAMS[r]
                parameters.aug_params.Jpeg.QF_min = JPEG_PARAMS[i]
                parameters.aug_params.Jpeg.QF_max = JPEG_PARAMS[i]
                parameters.aug_params.H264.QP_min = H264_PARAMS[i]
                parameters.aug_params.H264.QP_max = H264_PARAMS[i]

                if NUM_QUALITY_CLASSES == 1:
                    compression_transform = CustomJpegCompression(parameters.aug_params, inf_injection=None,
                                                            linear_map_factor=LINEAR_MAP_FACTOR, bin_map=BIN_MAP)
                    print("INFO -- current jpeg quality factor: {}".format(parameters.aug_params.Jpeg.QF_min))

                else:
                    compression_transform = CustomJpegCompression(parameters.aug_params, inf_injection=j,
                                                            linear_map_factor=LINEAR_MAP_FACTOR, bin_map=BIN_MAP)
                    print("INFO -- current jpeg quality factor: {}".format(parameters.aug_params.Jpeg.QF_min))


                transforms = [Normalize(),
                              Resolution(parameters.aug_params, parameters.sample_shape),
                              compression_transform,
                              ScaleToNetInputSize(sample_shape=parameters.sample_shape, aug_params=parameters.aug_params),
                              PerImageStandardizationTransform()]

                print("INFO -- current res factor: {}".format(parameters.aug_params.Resolution.min_res))

            else:
                transforms = [PerImageStandardizationTransform()]

            test_dataset = SyntheticDataGenerator(parameters.test_data_dir, model_params, mode='test',
                                                  transform=transforms)
            test_dataloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False,
                                         drop_last=False)

            timestamp = str(int(time.time()))
            prediction_file = os.path.join(pred_output_dir, "predictions-{}.csv".format(timestamp))

            # specify columns of csv file
            columns = (['predictions', 'license_numbers'])

            create_csv_file(prediction_file, columns)

            print("INFO -- start predicting to: {}".format(prediction_file))
            if CNN_BASED_ARCHITECTURE:
                baseline_predict_to_csv(prediction_file=prediction_file,
                                        dataloader=test_dataloader,
                                        h5_keys=None,
                                        model=model,
                                        alphabet=parameters.alphabet,
                                        batch_size=parameters.test_batch_size,
                                        device=parameters.device,
                                        max_pred_length=parameters.max_pred_length)
            else:
                predict_to_csv(prediction_file=prediction_file,
                               dataloader=test_dataloader,
                               h5_keys=None,
                               model=model,
                               alphabet=parameters.alphabet,
                               batch_size=parameters.test_batch_size,
                               device=parameters.device,
                               max_pred_length=parameters.max_pred_length)
            print("INFO -- Finished. Wrote output to {}".format(prediction_file))

print("INFO -- Finished testing on all directories!")
