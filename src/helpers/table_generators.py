import numpy as np
import pandas as pd
import re
import os


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def table_custom_transform(input_file: str, output_file: str, metric: str, transform_param: str) -> None:
    df = pd.read_csv(input_file)

    # extract transform parameter value from file name
    transform_param_value = np.array(list(map(lambda x: float(df.Filename.array[x].split("_")[-1].split(".csv")[0]), range(0, df.Filename.array.size))))
    # replace Filename with numeric param value
    df.Filename.replace(df.Filename.values, transform_param_value, inplace=True)
    # sort in ascending order
    df = df.sort_values(by=['Filename'])
    transform_param_value = np.sort(transform_param_value)
    row = []

    for i in range(transform_param_value.size):
        metric_val = df[metric].array[i]
        row.append([transform_param_value[i], metric_val])
    np_row = np.array(row)

    # create frame in right format
    frame = pd.DataFrame(np_row,
                         columns=[transform_param, metric])
    frame.to_csv(output_file, index=False)

# returns one metric for all training_configs (compression, width, noise)
def table_comp_res_noise(input_file: str, output_file: str, metric: str) -> None:
    df = pd.read_csv(input_file)

    dir_names = np.array(list(map(lambda x: os.path.basename(os.path.split(df.Filename.array[x])[0]), range(0, df.Filename.array.size))))
    row = []
    for i in range(dir_names.size):
        str_list = dir_names[i].split("_")
        target_width = str_list[3]
        noise_SNR_db = str_list[7]
        compression_rate = str_list[10]
        metric_val = df[metric].array[i]
        row.append([target_width, noise_SNR_db, compression_rate, metric_val])

    np_row = np.array(row)
    data_list = []
    for db in ['-2', '3', '20']:
        for jpeg_comp in ['95', '55', '30', '15', '1']:
            mask = (np.logical_and(np_row[:,1] == db, np_row[:,2] == jpeg_comp))
            filtered_row = np_row[mask]
            new_row = [db, jpeg_comp, -1, -1, -1, -1, -1]
            for i in range(filtered_row.shape[0]):
                width = filtered_row[i][0]
                pca = format(np.float(filtered_row[i][3]), '.4f')
                if width == '20':
                    new_row[2] = pca
                elif width == '50':
                    new_row[3] = pca
                elif width == '70':
                    new_row[4] = pca
                elif width == '125':
                    new_row[5] = pca
                else:
                    new_row[6] = pca
            data_list.append(tuple(new_row))

    # create frame in right format
    frame = pd.DataFrame(np.array(data_list), columns=['SNR in dB', 'JPEG quality factor', '20', '50', '70','125', '180'])
    frame.to_csv(output_file, index=False)


def table_res_jpeg(input_file: str, output_file:str, metric:str, jpeg_params: np.ndarray, res_params: np.ndarray) -> None:
    # open file holding evaluation results
    df = pd.read_csv(input_file)

    # init result matrix
    result_matrix = np.full((res_params.size, jpeg_params.size), np.inf)

    # extract dir names holding param information
    dir_names = np.array(
        list(map(lambda x: os.path.basename(os.path.split(df.Filename.array[x])[0]), range(0, df.Filename.array.size))))
    dir_names = sorted_alphanumeric(dir_names)

    # iterate over all parameters and fill matrix
    for i in range(res_params.size):
        for j in range(jpeg_params.size):
            result_matrix[i][j] = df.loc[df['Filename'].str.contains("/res_{}_jpeg_{}/".format(res_params[i], jpeg_params[j]))][metric].values[0]

    # add resolution parameters to matrix
    result_matrix = np.hstack((np.expand_dims(res_params, 1), result_matrix))

    # define column names for csv file and save
    column_names = ['res'] + list(map(lambda x: "jpeg_{}".format(x), jpeg_params))
    frame = pd.DataFrame(np.array(result_matrix),
                         columns=[column_names])
    frame.to_csv(output_file, index=False)


