import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from scipy.ndimage import center_of_mass
from pathlib import Path

def nearest(point, centers):
    near = 0
    min_dist = 10 ** 16

    for center in centers:
        dist = ((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) ** 0.5

        if dist < min_dist:
            min_dist = dist
            near = center

    return near

traectory_1 = []
traectory_2 = []
traectory_3 = []

point_1 = 0
point_2 = 0
point_3 = 0

for i in range(100):
    image = np.load(Path(__file__).parent / "out" / f"h_{i}.npy")

    labeled = label(image)
    centers = []
    for point in range(1, labeled.max() + 1):
         y, x = center_of_mass(labeled == point)
         centers.append((x, y))
    

    if i == 0:
        traectory_1.append(centers[0])
        traectory_2.append(centers[1])
        traectory_3.append(centers[2])

        point_1 = centers[0]
        point_2 = centers[1]
        point_3 = centers[2]
    else:
        position_1 = nearest(point_1, centers)
        position_2 = nearest(point_2, centers)
        position_3 = nearest(point_3, centers)

        traectory_1.append(position_1)
        traectory_2.append(position_2)
        traectory_3.append(position_3)

        point_1 = position_1
        point_2 = position_2
        point_3 = position_3
        
plt.figure(figsize=(10, 8))

x1 = [p[0] for p in traectory_1]
y1 = [p[1] for p in traectory_1]
plt.plot(x1, y1, 'r-o', markersize=4, linewidth=2, alpha=0.7)

x2 = [p[0] for p in traectory_2]
y2 = [p[1] for p in traectory_2]
plt.plot(x2, y2, 'b-o', markersize=4, linewidth=2, alpha=0.7)

x3 = [p[0] for p in traectory_3]
y3 = [p[1] for p in traectory_3]
plt.plot(x3, y3, 'g-o', markersize=4, linewidth=2, alpha=0.7)

plt.show()