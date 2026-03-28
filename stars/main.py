import numpy as np
from skimage.measure import label
from skimage.morphology import erosion
from pathlib import Path

image = np.load(Path(__file__).parent / "stars.npy")

new_image = erosion(image, np.ones((3, 3)))

amount = np.max(label(image)) - np.max(label(new_image))

print(amount)