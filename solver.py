#! /usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import get_model
from scripts.data_preprocess import get_data
from scripts.inference import inference_sudoku, norm
from scripts.validate_game import validate_solution

# Configuration du script
use_gpu = False
train_model = False
validate_model = False

input_dataset = "sudoku.csv"
load_model_location = "model/sudoku.model"
output_model_location = "model/sudoku-new.model"

if use_gpu:
    # Configure the GPU to use, if the hardware needs this
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(f"Chargement du jeu de données depuis {input_dataset}")
x_train, x_test, y_train, y_test = get_data(input_dataset)

if train_model:
    print("Entraînement du modèle")
    model = get_model()

    adam = keras.optimizers.Adam(lr=.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam)

    model.fit(x_train, y_train, batch_size=32, epochs=2)

    model.save(output_model_location, save_format='h5')
else:
    print(f"Chargement du modèle depuis {load_model_location}")
    model = keras.models.load_model(load_model_location)

if validate_model:
    def test_accuracy(feats, labels):
        correct = 0

        for i, feat in enumerate(feats):
            print(f"Test {i}\n", end="\t")

            pred = inference_sudoku(model, feat)

            true = labels[i].reshape((9, 9)) + 1

            if abs(true - pred).sum() == 0:
                correct += 1

        print(
            f"Accuracy: {correct / feats.shape[0]} ({correct} for {feats.shape[0]} tests)")


    print("Test et validation du modèle")
    test_accuracy(x_test[:100], y_test[:100])


def solve_sudoku(grid):
    grid = grid.replace('\n', '')
    grid = grid.replace(' ', '')
    grid = np.array([int(j) for j in grid]).reshape((9, 9, 1))
    grid = norm(grid)
    grid = inference_sudoku(model, grid)
    return grid


game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''

game = solve_sudoku(game)

print("Puzzle résolu :")
print(game)

validate_solution(game)

print("Fin")
