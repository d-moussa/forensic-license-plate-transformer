import csv
import os
import pandas as pd
from src.helpers.metrics import *


def create_csv_file(output_file, columns):
    # create csv file with given columns to test path
    with open(output_file, 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(columns)


def append_data_to_csv(csv_file, data):
    with open(csv_file, 'a') as csv:
        df = pd.DataFrame(data)
        df.to_csv(csv_file, mode='a', header=False, index=False)


def sort_csv_file(csv_file: str, sort_key: str):
    """sort csv by Directory and drop zero-column"""
    df = pd.read_csv(csv_file)
    df.sort_values(by=[sort_key], ignore_index=True, inplace=True)
    df = df.drop(columns=['Unnamed: 0'])
    f_sorted = os.path.basename(csv_file).split(".")[0] + '-sorted' + '.csv'
    df.to_csv(os.path.join(os.path.split(csv_file)[0], f_sorted))


def evaluate_from_csv(input_file, output_file, max_n=1):
    """

    :param input_file: file to read predictions from
    :param output_file: file to write metrics result to
    :param alphabet: alphabet of current sequence task
    :param max_chars: max. no of chars a label sequence can have in the test data
    :param max_n: top n accuracies to evaluate (only possible if present in input file)!
    :return:
    """
    # open input file and create output file
    df = pd.read_csv(input_file, sep=',')
    if os.path.exists(output_file):
        write_header = False
    else:
        write_header = True
    csv_output = open(output_file, 'a')

    # get predictions and labels from data frame
    if max_n > 1:
        top_keys = []
        for i in range(2, max_n + 1):
            top_keys.append("top_" + str(i))
        top_n_predictions = []
        top_n_predictions.append(df['predictions'])
        for k in top_keys:
            top_n_predictions.append(df[k])
        top_n_predictions = np.array(top_n_predictions)

    predictions = df['predictions']
    labels = df['license_numbers']

    # initialization
    acc = 0
    skipped_counter = 0
    levenshtein_dist = 0
    cer = 0

    if max_n > 1:
        top_2_to_n_accuracies = np.zeros((len(top_keys)))

    for i in range(predictions.size):
        # accuracy (of top 1 prediction)
        acc += accuracy(labels[i], predictions[i])

        # top 2 to n accuracy
        if max_n > 1:
            for top_acc in range(top_2_to_n_accuracies.shape[0]):
                top_2_to_n_accuracies[top_acc] += top_n_accuracy(labels[i], top_n_predictions[:top_acc + 2, i])

        # Levenshtein distance
        levenshtein_dist += levenshtein(labels[i], predictions[i])

        # Character error metric
        cer += character_error_rate(labels[i], predictions[i])

    # normalize metrics
    acc /= predictions.size

    if max_n > 1:
        top_2_to_n_accuracies /= predictions.size
    levenshtein_dist /= predictions.size
    cer /= predictions.size

    # generate dataframe and write results to file
    if max_n > 1:
        data_list = list([input_file, predictions.size, acc,
                          levenshtein_dist, cer,
                          (predictions.size - skipped_counter), ])
        columns = ['Filename', 'Samples in File', 'Accuracy',
                   'Levenshtein Distance', 'Character Error Rate']
        for k_idx in reversed(range(len(top_keys))):
            data_list.insert(3, top_2_to_n_accuracies[k_idx])
            columns.insert(3, top_keys[k_idx])
        df = pd.DataFrame(np.array(data_list).reshape(1, -1),
                          columns=columns)
    else:
        data_list = list([input_file, predictions.size, acc, levenshtein_dist, cer])
        df = pd.DataFrame(np.array(data_list).reshape(1, -1),
                          columns=['Filename', 'Samples in File', 'Accuracy', 'Levenshtein Distance',
                                   'Character Error Rate'])

    df.to_csv(csv_output, header=write_header)
    csv_output.close()

    print('accuracy: ' + str(acc))

    if max_n > 1:
        print('top 2 to n accuracies: ', top_2_to_n_accuracies)
    print('Levenshtein Distance: ' + str(levenshtein_dist))
    print('CER: ' + str(cer))
    print('evaluation data used: ', input_file)


def evaluate_on_dir(base_dir: str, result_file: str, file_pattern: str, max_n=1):
    """evaluates all prediction results saved in csv files in base_dir

    :param base_dir: directory which containes CSV prediction files to evaluate from
    :param result_file: path to output file to save results to
    :param alphabet: target vocabulary to use
    :param file_pattern: pattern to match CSV files that shall be evaluated
    :param max_n: top n predictions to evaluate
    :param max_chars: maximum label length occuring in data set
    """

    for f in os.listdir(base_dir):
        if os.path.isfile(os.path.join(base_dir, f)) and file_pattern in f:
            print("INFO -- Start evaluating on file: {}".format(f))
            evaluate_from_csv(input_file=os.path.join(base_dir, f), output_file=os.path.join(result_file), max_n=max_n)
