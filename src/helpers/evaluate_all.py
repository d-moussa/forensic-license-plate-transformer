import os
import sys
sys.path.append('src/utils/')
from alphabet import Alphabet
from csv_helpers import sort_csv_file, evaluate_on_dir
from table_generators import table_res_jpeg
import numpy as np

EVALUATE_PREDICTIONS = False
GENERATE_TABLES = True
INJECTION_CLASS_INPUT = False


#PATH = '/harddisk1/home/ca97siwo/inf4former/saved_models/baseline-effnet/predictions/'
#ALPHABET = Alphabet("/harddisk1/home/ca97siwo/inf4former/alphabets/german_lp_mapping_seq2seq.json")
PATH = '/home/moussa/inf4former/saved_models/inf4former_jpeg_res_100_classes_bilinear/predictions/Low_Res/'
ALPHABET = Alphabet("/home/moussa/inf4former/alphabets/german_lp_mapping_seq2seq.json")
# ALPHABET = Alphabet("/home/moussa/lp_recognition/alphabets/czech_lp_mapping_seq2seq.json")
PATTERN = '.csv'
print("INFO -- matching on pattern: {}".format(PATTERN))

def eval_subdirs(result_file, root_path):
    # get all subdirectiories containing test data
    test_data_dirs = os.listdir(root_path)

    for test_dir in test_data_dirs:

        if os.path.basename(test_dir) == 'results': continue
        #if os.path.basename(test_dir) != 'res_15_jpeg_1': continue
        print("INFO -- Start evaluating on dir: {}".format(os.path.join(root_path,test_dir)))
        evaluate_on_dir(base_dir=os.path.join(root_path, test_dir), result_file=result_file, file_pattern=PATTERN,
                        max_n=1)

if EVALUATE_PREDICTIONS:
    if not INJECTION_CLASS_INPUT:
        exp_runs = os.listdir(PATH)
        result_dir = os.path.join(PATH, 'results')
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, 'results.csv')
        eval_subdirs(result_file, root_path=PATH)
        sort_csv_file(result_file, 'Filename')
    else:
        exp_runs = os.listdir(PATH)
        for dir in exp_runs:
            base_dir = os.path.join(PATH, dir)
            result_dir = os.path.join(base_dir, 'results')
            os.makedirs(result_dir, exist_ok=True)
            result_file = os.path.join(result_dir, 'results.csv')
            eval_subdirs(result_file=result_file, root_path=base_dir)
            sort_csv_file(result_file, 'Filename')



if GENERATE_TABLES:
    path_b = PATH #'/home/moussa/inf4former/saved_models/baseline-cnn/predictions/Intervallabtastung/'
    table_res_jpeg(os.path.join(path_b, 'results/results-sorted.csv'),
                   output_file=os.path.join(path_b, 'result_mat.csv'),
                   metric='Accuracy',
                   jpeg_params= np.concatenate((np.arange(1,100,4), [100])),
                   res_params=np.arange(20, 26, 1))
