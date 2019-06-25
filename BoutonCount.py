import NetworkTrain
import CellFromIllustrator
import os
import argparse
import scipy.ndimage
import numpy as np
from PIL import Image
import time
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


class CreateSVG():
    def __init__(self, output_svgname, artboard_size_xy, input_image):
        self.output_svgname = output_svgname
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
        with open(self.output_svgname, 'w+') as f:
            f.write(self.string)


def count_brain(brain_directory, model):
    load_time = 0
    proposal_time = 0
    ML_time = 0
    output_time = 0
    for directory in sorted(os.listdir(brain_directory)):
        directory = os.path.join(brain_directory, directory)
        output_jpg = "%s.jpg" % directory
        output_svg = "%s.svg" % directory
        if not os.path.isdir(os.path.join(brain_directory, directory)):
            continue
        if os.path.exists(output_svg):
            continue
        start_time = time.time()
        if os.path.exists(output_jpg):
            arr = CellFromIllustrator.file_to_array(output_jpg).astype(np.float32)
        else:
            arrs = []
            dirs = [f for f in sorted(os.listdir(directory)) if os.path.isfile(os.path.join(directory, f))]
            if len(dirs) == 0:  # If exported through CellSens
                for r, d, f in os.walk(directory):
                    for file in f:
                        if file.endswith(".tif") and "EFI" in file:
                            print(os.path.join(r, file))
                            arrs.append(
                                CellFromIllustrator.file_to_array(os.path.join(r, file)).astype(np.float32))
            else:
                try:
                    for file in dirs:  # If exported through VS-ASW
                        if file.endswith(".tif") and "IMAGE" in file:
                            arrs.append(CellFromIllustrator.file_to_array(os.path.join(directory, file)).astype(np.float32))
                except OSError:
                    print("Images for %s do not exist" % directory)
                    continue
            print("Working on %s" % output_jpg)
            if arrs[0].shape[0] > arrs[0].shape[1]:
                arr = np.concatenate(arrs, axis=1)
            else:
                arr = np.concatenate(arrs, axis=0)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        im = Image.fromarray(arr * 255).convert("L")
        im.save(output_jpg)
        if os.path.exists(output_svg):
            continue
        print("Working on %s" % output_svg)
        load_time += time.time() - start_time
        start_time = time.time()
        X, COMs = local_maxima_generate_points(arr, find_maxima=False)
        proposal_time += time.time() - start_time
        start_time = time.time()
        output = model.predict(X)
        ML_time += time.time() - start_time
        start_time = time.time()
        svg = CreateSVG(output_svg, arr.shape, output_jpg)
        true_cells = COMs[output[:, 0] > output[:, 1], :]
        for i in range(true_cells.shape[0]):
            svg.add_symbol(location_xy=(true_cells[i, 1], true_cells[i, 0]))
        svg.output()
        output_time += time.time() - start_time
    print("Load Time: %.2f" % load_time)
    print("Proposal Time: %.2f" % proposal_time)
    print("ML Time: %.2f" % ML_time)
    print("Output Time: %.2f" % output_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This program uses the pretrained NN to search for cells")
    parser.add_argument("-d", "--directory", type=str, help="Brain level directory")
    parser.add_argument("-o", "--order", type=str, help="Order file for multiple brains")
    args = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    model = NetworkTrain.load_json_model("Boutons")
    if args.directory is not None:
        count_brain(args.directory, model)
    elif args.order is not None:
        dir_name = os.path.dirname(args.order)
        with open(args.order) as f:
            for line in f:
                line = os.path.join(dir_name, line.strip())
                count_brain(line, model)
