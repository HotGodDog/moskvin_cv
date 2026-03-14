import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import opening

image = np.load("wires/wires6.npy")
struct = np.ones((3, 1))

labeled_image = label(image)

for i in range(1, np.max(labeled_image) + 1):
    print(f"Original {i}")
    process = opening(labeled_image == i, struct)
    labeled_process = label(process)
    print(f"Processed {np.max(labeled_process)}")
    print()

process = opening(image, struct)

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(process)
plt.show()
