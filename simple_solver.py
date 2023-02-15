#! /usr/bin/env python3

import numpy as np
from tensorflow import keras
from scripts.inference import inference_sudoku, norm
from scripts.validate_game import validate_solution

input_dataset = "sudoku.csv"
load_model_location = "model/sudoku.model"
output_model_location = "model/sudoku-new.model"

print(f"Chargement du modèle depuis {load_model_location}")
model = keras.models.load_model(load_model_location)


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

game = input(pyCells)

game = solve_sudoku(game)

print("Puzzle résolu :")
print(game)

validate_solution(game)

print("Fin")
