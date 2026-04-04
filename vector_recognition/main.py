import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path


save_path = Path(__file__).parent

def horizontal_symmetry(region):
    image = region.image
    h, w = image.shape
    top = image[:h//2, :]
    bottom = image[h//2:, :]
    min_h = min(top.shape[0], bottom.shape[0])
    top = top[-min_h:, :]  
    bottom = bottom[:min_h, :] 
    bottom_flipped = np.flipud(bottom)
    intersection = np.sum(top & bottom_flipped)
    union = np.sum(top | bottom_flipped)
    if union == 0:
        return 1.0
    return intersection / union

def vertical_symmetry(region):
    image = region.image
    h, w = image.shape
    left = image[:, :w//2]
    right = image[:, w//2:]
    min_w = min(left.shape[1], right.shape[1])
    left = left[:, -min_w:]
    right = right[:, :min_w]
    left_flipped = np.fliplr(left)
    intersection = np.sum(left_flipped & right)
    union = np.sum(left_flipped | right)
    if union == 0:
        return 1.0
    return intersection / union

def count_endpoints(region):
    image = region.image
    endpoints = 0
    h, w = image.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if image[y, x]:
                neighbors = sum([
                    image[y-1, x], image[y+1, x],
                    image[y, x-1], image[y, x+1],
                    image[y-1, x-1], image[y-1, x+1],
                    image[y+1, x-1], image[y+1, x+1]
                ])
                if neighbors == 1:
                    endpoints += 1
    return endpoints / (image.sum() + 1)

def calculate_compactness(region):
    area = region.area
    perimeter = region.perimeter
    if perimeter == 0 or area == 0:
        return 0
    compactness = (4 * np.pi * area) / (perimeter * perimeter)
    return compactness

def count_crosses(region):
    image = region.image
    crosses = 0
    h, w = image.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if image[y, x]:
                neighbors = sum([
                    image[y-1, x], image[y+1, x],
                    image[y, x-1], image[y, x+1],
                    image[y-1, x-1], image[y-1, x+1],
                    image[y+1, x-1], image[y+1, x+1]
                ])
                if neighbors >= 4:
                    crosses += 1
    return crosses / (image.sum() + 1)

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)
    vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
    hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum()
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]

    crosses = count_crosses(region)
    endpoints = count_endpoints(region)
    compactness = calculate_compactness(region)
    ver_sym = vertical_symmetry(region)
    hor_sym = horizontal_symmetry(region)
    
    return np.array([region.area / region.image.size, 
                     cy, cx, 
                     perimeter, 
                     holes, 
                     vlines, hlines, 
                     eccentricity,
                     aspect,
                     crosses,
                     endpoints, 
                     compactness,
                     ver_sym, 
                     hor_sym])

weights = np.array([
    1.0,   # area
    1.0,   # cy
    0.5,   # cx
    0.8,   # perimeter
    12.0,  # holes     
    6.0,   # vlines
    3.0,   # hlines
    2.0,   # eccentricity
    8.0,   # aspect 
    12.0,  # crosses  
    18.0,  # endpoints
    2.0,   # compactness
    12.0,  # ver_sym
    5.0    # hor_sym
])

def classificator(region, templates_norm, mean, std, weights):
    feat = extractor(region)
    feat_norm = (feat - mean) / std
    best_sym = None
    best_dist = float('inf')
    for sym, tmpl in templates_norm.items():
        diff = tmpl - feat_norm
        diff = diff * weights
        dist = np.linalg.norm(diff)
        if dist < best_dist:
            best_dist = dist
            best_sym = sym
    return best_sym

template = imread(save_path / "alphabet-small.png")[:, :, :-1]
template = template.sum(2)
binary = template != 765.

labeled = label(binary)
props = regionprops(labeled)

templates = {}

for region, symbol in zip(props, ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]):
    templates[symbol] = extractor(region)

all_feat = np.array(list(templates.values()))
mean = all_feat.mean(axis=0)
std = all_feat.std(axis=0)
std[std == 0] = 1

templates_norm = {}
for sym, f in templates.items():
    templates_norm[sym] = (f - mean) / std

image = imread(save_path / "alphabet.png")[:, :, :-1]
abinary = image.mean(2) > 0
alphabet = label(abinary)
aprops = regionprops(alphabet)
result = {}

image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))

for region in aprops:
    symbol = classificator(region, templates_norm, mean, std, weights)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(result)
plt.show()