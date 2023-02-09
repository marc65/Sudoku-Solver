import copy

import numpy as np
from tensorflow.keras.models import Sequential


def norm(a: np.numarray) -> np.numarray:
    return (a / 9) - .5


def denorm(a: np.numarray) -> np.numarray:
    return (a + .5) * 9


def inference_sudoku(model: Sequential, sample: np.numarray) -> np.numarray:
    """
        This function solves the sudoku by filling blank positions one by one,
        keeping the highest confidence one from the network’s answer.
    """

    feat = copy.copy(sample)

    while 1:

        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if mask.sum() == 0:
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)

    return pred
