import NetworkTrain
import CellFromIllustrator
import os
import argparse
import scipy.ndimage
import numpy as np
from PIL import Image
import time
import shutil
from Constants import SIZE, RADIUS

class OutputFileType:
    def __init__(self):
        self.svg = False
        self.csv = False
        self.jpg = False
        
    def any(self):
        return self.svg or self.csv or self.jpg

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


class CreatePNG():
    def __init__(self, output_bouton_png_name, artboard_size_xy, downscale=1.0):
        self.output_name = output_bouton_png_name
        self.x = artboard_size_xy[0]
        self.y = artboard_size_xy[1]
        self.arr = np.zeros(shape=(self.x, self.y), dtype=np.bool)
        self.radius = 5
        self.symbol_shape = np.zeros((self.radius * 2 + 1, self.radius * 2 + 1), dtype=np.bool)
        self.downscale = downscale
        for x in range(self.radius * 2 + 1):
            for y in range(self.radius * 2 + 1):
                if (x - self.radius) ** 2 + (y - self.radius) ** 2 <= self.radius ** 2:
                    self.symbol_shape[x, y] = True
                    
    def add_symbol(self, location_xy):
        try:
            y, x = location_xy[0], location_xy[1]
            self.arr[x - self.radius: x + self.radius + 1, y - self.radius: y + self.radius + 1] = \
                np.logical_or(self.arr[x - self.radius: x + self.radius + 1, y - self.radius: y + self.radius + 1], self.symbol_shape)
        except ValueError as e:
            pass

    def output(self):
        image_array = np.zeros((self.arr.shape[0], self.arr.shape[1], 4), dtype=np.uint8)
        image_array[:, :, 0] = 255  # Convert to image array
        image_array[:, :, 1] = 255 - 255 * self.arr
        image_array[:, :, 2] = 255 - 255 * self.arr
        image_array[:, :, 3] = self.arr * 255
        image = Image.fromarray(image_array, 'RGBA')
        image = image.resize((int(self.arr.shape[0] * self.downscale), int(self.arr.shape[1] * self.downscale)), Image.BILINEAR)
        image.save(self.output_name)


class CreateSVG():
    def __init__(self, output_svgname, artboard_size_xy, input_image):
        self.output_svgname = output_svgname
        x = artboard_size_xy[1]
        y = artboard_size_xy[0]
        self.strings = ["""
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="%s" height="%s" viewBox="0 0 %s %s">
  <defs>
    <style>
      .cls-1 {
        isolation: isolate;
      }
    </style>
    <symbol id="Bouton" data-name="Bouton" viewBox="0 0 1 1">
      <circle cx="0.5" cy="0.5" r="0.5"/>
    </symbol>
  </defs>
  <image class="cls-1" x="0.5" y="0.5" width="%s" height="%s" xlink:href="%s"/>
    """ % (x, y, y, x, y, x, input_image)]

    def add_symbol(self, location_xy):
        self.strings.append(
            """  <use width="1" height="1" transform="translate(%s %s)" xlink:href="#Bouton"/>
            """ % (location_xy[0], location_xy[1]))

    def output(self):
        self.strings.append(\
"""</svg>""")
        string = "".join(self.strings)
        with open(self.output_svgname, 'w+') as f:
            f.write(string)


def generate_border_and_mask(mask_file_path):
    if mask_file_path is None:
        return None, None
    else:
        mask = CellFromIllustrator.file_to_array(mask_file_path)
        mask = mask > 128
        border = np.bitwise_and(mask, np.invert(scipy.ndimage.morphology.binary_erosion(mask)))
        return mask, border


def count_section(directory, model, oft):
    input_mask = os.path.join(directory, "Mask.jpg")
    output_image = os.path.join(directory, "Image.jpg")
    output_svg = os.path.join(directory, "Boutons.svg")
    output_png = os.path.join(directory, "Boutons.png")
    output_csv = os.path.join(directory, "Boutons.csv")
    arr = None
    start_time = time.time()
    if os.path.exists(output_image):
        arr = CellFromIllustrator.file_to_array(output_image).astype(np.float32)
    else:
        files = [f for f in sorted(os.listdir(directory)) if
                 os.path.isfile(os.path.join(directory, f)) and (f.endswith(".tif") or f.endswith("png")) and "IMAGE" in f]
        if len(files) == 0:  # If exported through CellSens
            for r, d, f in os.walk(directory, topdown=False):
                for file in f:
                    if file.endswith(".tif") and "EFI" in file:
                        print(os.path.join(r, file))
                        arr = CellFromIllustrator.file_to_array(os.path.join(r, file)).astype(np.float32)
                        os.remove(os.path.join(r, file))
                if r != directory:
                    os.rmdir(r)
        else:
            try:
                arrs = []
                for file in files:  # If exported through VS-ASW
                    arrs.append(CellFromIllustrator.file_to_array(os.path.join(directory, file)).astype(np.float32))
                    os.remove(os.path.join(directory, file))
                if arrs[0].shape[0] > arrs[0].shape[1]:
                    arr = np.concatenate(arrs, axis=1)
                else:
                    arr = np.concatenate(arrs, axis=0)
            except OSError:
                print("Images for %s do not exist" % directory)
                return
        print("Working on %s" % output_image)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    im = Image.fromarray(arr * 255).convert("L")
    im.save(output_image)
    if oft.any():
        mask = None
        if os.path.exists(input_mask):
            mask, _ = generate_border_and_mask(input_mask)
        print("Working on %s" % output_svg)
        print("Load Time: %.2f" % (time.time() - start_time))
        start_time = time.time()
        X, COMs = local_maxima_generate_points(arr, mask, find_maxima=False)
        print("Proposal Time: %.2f" % (time.time() - start_time))
        start_time = time.time()
        output = model.predict(X)
        print("ML Time: %.2f" % (time.time() - start_time))
        start_time = time.time()
        true_cells = COMs[output[:, 0] > 0.95, :]
        if oft.svg and not os.path.exists(output_svg):
            svg = CreateSVG(output_svg, arr.shape, output_image)
            for i in range(true_cells.shape[0]):
                svg.add_symbol(location_xy=(true_cells[i, 1], true_cells[i, 0]))
            svg.output()
        if oft.jpg:
            jpg = CreatePNG(output_png, arr.shape, downscale=0.3)
            for i in range(true_cells.shape[0]):
                jpg.add_symbol(location_xy=(true_cells[i, 1], true_cells[i, 0]))
            jpg.output()
        if oft.csv:
            str = ""
            for i in range(true_cells.shape[0]):
                str += "%s,%s%s" % (true_cells[i, 1], true_cells[i, 0], os.linesep)
            with open(output_csv, 'w') as f:
                f.write(str)
        print("Output Time: %.2f" % (time.time() - start_time))


def count_brain(brain_directory, model, oft):
    for directory in sorted(os.listdir(brain_directory)):
        directory = os.path.join(brain_directory, directory)
        if os.path.isdir(directory):
            count_section(directory, model, oft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program uses the pretrained NN to search for cells or boutons")
    parser.add_argument("-d", "--directory", type=str, help="Brain level directory")
    parser.add_argument("-o", "--order", type=str, help="Order file for multiple brains")
    parser.add_argument("-f", "--folder", type=str, help="File for one section from one brain")
    parser.add_argument("-j", "--jpg", action="store_true", help="Output jpg file Boutons.tif")
    parser.add_argument("-s", "--svg", action="store_true", help="Output svg file Boutons.svg")
    parser.add_argument("-c", "--csv", action="store_true", help="Output csv file Boutons.csv")
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    model = NetworkTrain.load_json_model("Boutons")
    oft = OutputFileType()
    oft.svg, oft.jpg, oft.csv = args.svg, args.jpg, args.csv
    if args.directory is not None:
        count_brain(args.directory, model, oft)
    elif args.order is not None:
        dir_name = os.path.dirname(args.order)
        with open(args.order) as f:
            for line in f:
                line = os.path.join(dir_name, line.strip())
                count_brain(line, model, oft)
    elif args.folder is not None:
        count_section(args.folder, model, oft)
