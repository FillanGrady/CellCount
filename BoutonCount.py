import NetworkTrain
import CellFromIllustrator
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.segmentation
import scipy.ndimage
import numpy as np
import math
from PIL import Image
from GenerateTrainData import data_augmentation
from Constants import SIZE, RADIUS


def get_next_number(folder):
    i = 0
    while True:
        i += 1
        if not os.path.exists(os.path.join(folder, "%02d.npz" % i)):
            break
    return os.path.join(folder, "%02d" % i)


def detect_local_maxima(arr, mask=None):
    scale = 4
    dilated = scipy.ndimage.morphology.grey_dilation(arr, size=(scale, scale))
    eroded = scipy.ndimage.morphology.grey_erosion(arr, size=(scale, scale))
    x1 = arr == dilated
    x2 = arr > eroded
    x = np.logical_and(x1, x2)
    if mask is not None:
        x = x * mask
    x[:RADIUS, :] = False
    x[-RADIUS:, :] = False
    x[:, :RADIUS] = False
    x[:, -RADIUS:] = False
    return np.where(x)


def detect_local_minima(arr, mask=None):
    scale = 4
    dilated = scipy.ndimage.morphology.grey_dilation(arr, size=(scale, scale))
    eroded = scipy.ndimage.morphology.grey_erosion(arr, size=(scale, scale))
    x1 = arr == eroded
    x2 = arr < dilated - .01
    x = np.logical_and(x1, x2)
    if mask is not None:
        x = x * mask
    x[:RADIUS, :] = False
    x[-RADIUS:, :] = False
    x[:, :RADIUS] = False
    x[:, -RADIUS:] = False
    return np.where(x)


def local_maxima_generate_points(arr, mask=None, find_maxima=True):
    float_arr = arr.astype(np.float32)
    scale = 1
    fuzzy = scipy.ndimage.gaussian_filter(float_arr, sigma=scale)
    if find_maxima:
        maxima = np.array(detect_local_maxima(fuzzy, mask)).T
    else:
        maxima = np.array(detect_local_minima(fuzzy, mask)).T
    assert maxima.shape[0] > 0
    X = np.zeros(shape=(maxima.shape[0], 1, SIZE, SIZE), dtype=np.float32)
    for i in range(maxima.shape[0]):
        x, y = maxima[i, :]
        X[i, 0, :, :] = arr[x - RADIUS:x + RADIUS, y - RADIUS:y + RADIUS]
    return X, maxima


def felzenszwalb_generate_points(arr):
    fuzzy = scipy.ndimage.filters.gaussian_filter(arr, sigma=10)
    felzenszwalb = skimage.segmentation.felzenszwalb(arr, scale=.1, sigma=0.8, min_size=100)

    coms = scipy.ndimage.measurements.center_of_mass(arr, felzenszwalb, range(1, np.max(felzenszwalb)))
    X = []
    COMs = []
    for com in coms:
        if math.isnan(com[0]) or math.isnan(com[1]):
            continue
        x = int(com[0])
        y = int(com[1])
        if arr[x, y] > fuzzy[x, y]:
            x, y = CellFromIllustrator.wobble_point(fuzzy, x, y)
            sub_arr = arr[x-RADIUS:x+RADIUS, y-RADIUS:y+RADIUS]
            if sub_arr.shape == (SIZE, SIZE):
                X.append(sub_arr)
                COMs.append((x, y))
    X = np.array(X).reshape(-1, 1, 80, 80)
    COMs = np.array(COMs)
    return X, COMs


def take_sub_array(arr, x1, x2, y1, y2):
    place_x1 = 0
    place_x2 = x2 - x1
    place_y1 = 0
    place_y2 = y2 - y1
    sub_array = np.zeros(shape=(x2 - x1, y2 - y1))
    if x1 < 0:
        place_x1 = -x1
        x1 = 0
    if y1 < 0:
        place_y1 = -y1
        y1 = 0
    if x2 > arr.shape[0]:
        place_x2 = arr.shape[0] - x1
        x2 = arr.shape[0]
    if y2 > arr.shape[1]:
        place_y2 = arr.shape[1] - y1
        y2 = arr.shape[1]
    sub_array[place_x1: place_x2, place_y1: place_y2] = arr[x1: x2, y1: y2]
    return sub_array


class CreateSVG():
    def __init__(self, output_filename, artboard_size_xy, input_image):
        self.output_filename = output_filename
        x = artboard_size_xy[1]
        y = artboard_size_xy[0]
        self.string = \
"""<?xml version="1.0" encoding="utf-8"?>
<!-- Generator: Adobe Illustrator 23.0.3, SVG Export Plug-In . SVG Version: 6.00 Build 0)  -->
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 %s %s" style="enable-background:new 0 0 %s %s;" xml:space="preserve">
<style type="text/css">
	.st0{fill:#FF0000;}
</style>
<symbol  id="Bouton" viewBox="-1.9 -1.9 3.8 3.8">
	<circle class="st0" cx="0" cy="0" r="1.9"/>
</symbol>
<g id="Image">
	<g id="_x30_2.tif_1_">
		<image style="overflow:visible;" width="%s" height="%s" id="_x30_2" xlink:href="%s" >
		</image>
	</g>
</g>
<g id="Boutons">""" % (x, y, x, y, x, y, input_image)

    def add_symbol(self, location_xy):
        self.string += \
"""		<use xlink:href="#Bouton"  width="3.8" height="3.8" x="-1.9" y="-1.9" transform="matrix(1.0026 0 0 -1.0026 %s %s)" style="overflow:visible;enable-background:new    ;"/>
""" % (location_xy[0], location_xy[1])

    def output(self):
        self.string += \
"""</g>
</svg>"""
        with open(self.output_filename, 'w+') as f:
            f.write(self.string)


if __name__ == "__main__":
    mng = plt.get_current_fig_manager()

    parser = argparse.ArgumentParser(description="This program uses the pretrained NN to search for cells")
    parser.add_argument("-d", "--directory", type=str, help="Top level directory")
    args = parser.parse_args()

    model = NetworkTrain.load_json_model("Boutons")
    for directory in os.listdir(args.directory):
        print(os.path.join(args.directory, directory))
        if not os.path.isdir(os.path.join(args.directory, directory)):
            continue
        arrs = []
        for file in os.listdir(os.path.join(args.directory, directory)):
            if file.endswith(".tif"):
                arrs.append(CellFromIllustrator.file_to_array(os.path.join(args.directory, directory, file)).astype(np.float32))
        arr = np.concatenate(arrs, axis=1)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        im = Image.fromarray(arr * 255).convert("L")
        image_path = os.path.join(args.directory, "%s.jpg" % directory)
        im.save(image_path)
        X, COMs = local_maxima_generate_points(arr, find_maxima=False)
        output = model.predict(X)
        prediction = output[:, 0] > output[:, 1]
        svg = CreateSVG(os.path.join(args.directory, "%s.svg" % directory), arr.shape, image_path)
        for i in range(prediction.size):
            if prediction[i]:
                x, y = COMs[i]
                svg.add_symbol(location_xy=(y, x))
        svg.output()
