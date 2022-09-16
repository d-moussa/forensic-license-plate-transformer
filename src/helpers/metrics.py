import numpy as np
from nltk.metrics import edit_distance

def accuracy(y_true, y_pred):
    """
    evaluates whether the predicted sequence is correct or not
    :param y_true: the groundtruth for a sample (label)
    :param y_pred: the predicted sequence for a sample
    :return: 1 if y_pred equals y_true, 0 otherwise
    """
    if y_pred == y_true:
        return 1
    else:
        return 0

def accuracy_per_character(y_true, y_pred, max_chars):
    """
    evaluates the accuracy per character (whether character is predicted correctly or not). Samples longer than
    max_chars_per_string are skipped, and y_true and y_pred must have the same shape
    :param y_true:  the groundtruth for a sample (label)
    :param y_pred:  the predicted sequence for a sample
    :return: 1D-array with one field per character c or None.
            Each fields contains info whether c was predicted correctly (1) or not (0)
            if input sequences are not of equal length, None is returned
    """
    accuracy = np.ones((max_chars))
    diff = len(y_true) - len(y_pred)
    # metric only works on sequences of same length
    if diff == 0:
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                accuracy[i] = 0
    else:
        return None

    return np.array(accuracy)

def accuracy_per_alphabet_character(y_true, y_pred, alphabet):
    """
        evaluates the accuracy per alphabet character
        :param y_true:  the groundtruth for a sample (label)
        :param y_pred:  the predicted sequence for a sample
        :return: (36, 2)-array a. Axis 0 represents the characters in the alphabet in the same order. Axis 1 shows the
         correct predictions and total occurences of the digits. E.g. a[0][0] returns the number of correct predictions
         of character 'A' in y_pred and a[0][1] returns the no. of total occurrences of char 'A' in y_true
        """

    # initialization
    a = np.zeros((alphabet.size, 2))
    diff = len(y_true) - len(y_pred)

    # metric only works on sequences of same length
    if diff == 0:
        for i in range(len(y_true)):
            # get index of character label in alphabet
            index = np.where(np.array(alphabet.units) == y_true[i])[0][0]
            # count label and correct predictions
            if y_true[i] != y_pred[i]:
                a[index][1] += 1
            else:
                a[index] += 1
        return a

    else:
        return None

def levenshtein(y_true, y_pred):
    """
            computes the levenshtein distance of two strings
            :param y_true:  the groundtruth for a sample (label)
            :param y_pred:  the predicted sequence for a sample
            :return: scalar leveshtein distance
            """
    return edit_distance(y_true, y_pred, substitution_cost=1, transpositions=False)


def character_error_rate(y_true, y_pred):
    """
                computes the character error rate (CER) of two strings
                :param y_true:  the groundtruth for a sample (label)
                :param y_pred:  the predicted sequence for a sample
                :return: scalar (CER)
                """
    levenshtein_dist = edit_distance(y_true, y_pred, substitution_cost=1, transpositions=False)
    return levenshtein_dist / len(y_true)


def top_n_accuracy(y_true, y_preds):
    if y_true in y_preds:
        return 1
    else:
        return  0
