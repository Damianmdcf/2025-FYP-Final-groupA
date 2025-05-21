import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask(img_gray, threshold=40):
    seed = np.unravel_index(np.argmin(img_gray), img_gray.shape)
    h, w = img_gray.shape
    mask = np.zeros((h, w), dtype=bool)
    region_pixels = []
    stack = [seed]

    region_mean = float(img_gray[seed])
    count = 1

    mask[seed] = True
    region_pixels.append(seed)

    neighbors = [(-1, 0), (1,0), (0,-1), (0,1)]

    while len(stack) > 0:
        (x, y) = stack[-1]
        stack.pop()

        for (dx, dy) in neighbors:
            nx = x + dx
            ny = y + dy

            if (0 <= nx <= h-1) and (0 <= ny <= w-1) and (mask[nx, ny] == 0):
                pixel_val = float(img_gray[nx, ny])

                if abs(pixel_val - region_mean) <= threshold:
                    mask[nx, ny] = True
                    stack.append((nx, ny))
                    region_pixels.append((nx, ny))

                    region_mean = (region_mean * count + pixel_val) / (count + 1)
                    count += 1

    return mask
