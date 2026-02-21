import matplotlib.pyplot as plt
import numpy as np

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")
assert image.shape[0] == image.shape[1]

color1 = [0, 128, 255]
color2 = [255, 128, 0]

for i in range(size):
    for j in range(size):
        t = (i + j) / (2 * (size - 1))
        r = lerp(color1[0], color2[0], t)
        g = lerp(color1[1], color2[1], t)
        b = lerp(color1[2], color2[2], t)
        image[i, j] = [r, g, b]

plt.figure(1)
plt.imshow(image)
plt.show()
