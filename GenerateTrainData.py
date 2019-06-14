import argparse
import numpy as np
import CellFromIllustrator
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from Constants import SIZE, RADIUS


def read_csv_file(csv_file_name):
    lines = []
    with open(csv_file_name, "r") as f:
        for line in f:
            lines.append(line.strip().split(","))
    lines.sort(key=lambda x: x[0])
    return lines


def data_augmentation(X, Y, rotate=False, repeat=1):
    """
    Allows for data augmentation to increase the training set
    """
    Xs = [X, X[::-1, :], X[:, ::-1], X[::-1, ::-1]]
    if rotate:
        Xs.append(np.rot90(X))
        Xs.append(np.rot90(X, k=3))
    Xs = Xs * repeat
    Ys = [Y] * len(Xs)
    return Xs, Ys


def generate_negative_data(arr, cell_map, number):
    """
    Creates negative training data for training the network
    Creates NUMBER (x,y) pairs
    x is an array of size SIZE*SIZE drawn at random from arr, weighted by the intensity of that pixel in arr
    If x overlaps too much with the user selected positive cells from cell_map, throw it out
    :param arr:
    :param cell_map:
    :param number:
    :return:
    """
    X = []
    Y = []
    probabilities = (255 - np.ravel(arr).astype(np.uint64)) ^ 3
    probabilities = probabilities / probabilities.sum()
    start_locations = np.random.choice(arr.size, size=number, p=probabilities)
    # Generates a weighted random vector NUMBER long, where the probability of any value is the lightness of that pixel
    xs, ys = np.unravel_index(start_locations, arr.shape)
    for x, y in zip(xs, ys):

        if np.average(cell_map[int(x - RADIUS): int(x + RADIUS), int(y - RADIUS): int(y + RADIUS)]) < 0.5:
            sub_arr = arr[int(x - RADIUS): int(x + RADIUS), int(y - RADIUS): int(y + RADIUS)]
            if sub_arr.size == SIZE ** 2:
                X.append(sub_arr)
                Y.append([0, 1])
    return X, Y


def generate_data(lines):
    X = []
    Y = []
    last_file = lines[0][0]
    arr = CellFromIllustrator.file_to_array(lines[0][0]).astype(np.uint8)
    cell_map = np.zeros(arr.shape, dtype=np.bool)
    for line in lines:
        if line[0] != last_file:
            X_, Y_ = generate_negative_data(arr, cell_map, 5000)
            X.extend(X_)
            Y.extend(Y_)
            arr = CellFromIllustrator.file_to_array(line[0]).astype(np.uint8)
            last_file = line[0]
            cell_map = np.zeros(arr.shape, dtype=np.bool)

            """
            We need to generate negative training data, but we don't want the negative "no cell" images to accidentally
            overlap with the cells.  So every cell is marked as True, and everything else is marked as False
            """
        if len(line) > 7:
            x = int(line[4])
            y = int(line[5])
            w = int(line[6])
            h = int(line[7])
            subset = arr[x: x + w, y: y + h]
            if subset.shape != (SIZE, SIZE):
                continue

            Xs, Ys = data_augmentation(subset, [1, 0])
            X.extend(Xs)
            Y.extend(Ys)

            cell_map[x: x + w, y: y + h] = True
    X_, Y_ = generate_negative_data(arr, cell_map, 5000)
    X.extend(X_)
    Y.extend(Y_)
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NN")
    parser.add_argument("-c", "--csv", type=str, help="Filepath of input csv file")
    parser.add_argument("-t", "--train", type=str, help="Directory to output training data")
    args = parser.parse_args()
    lines = read_csv_file(args.csv)
    X, Y = generate_data(lines)
    X = np.array(X)
    Y = np.array(Y)
    if not os.path.isdir(args.train):
        os.mkdir(args.train)
    np.savez_compressed(os.path.join(args.train, "00"), X=X, Y=Y)