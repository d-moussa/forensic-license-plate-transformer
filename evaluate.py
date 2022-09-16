import os
from src.helpers.csv_helpers import sort_csv_file, evaluate_on_dir
import argparse


parser = argparse.ArgumentParser(description='Evaluates predictions and writes results to file')
parser.add_argument('input_dir', type=str, help='path to CSV prediction file')
args = parser.parse_args()

RESULT_DIR = os.path.join(args.input_dir, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULT_DIR, 'results.csv')
PATTERN = '.csv'
print("INFO -- matching on pattern: {}".format(PATTERN))

evaluate_on_dir(base_dir=args.input_dir, result_file=RESULT_FILE, file_pattern=PATTERN, max_n=1)
print("INFO -- Finished testing on all Files")
sort_csv_file(RESULT_FILE, 'Filename')
