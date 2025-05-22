import numpy as np
import cv2

def get_mask(img_rgb, img_th1=None, threshold=40):
    using_img_th1 = True

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if img_th1 is None:
        using_img_th1 = False
        seed = np.unravel_index(np.argmin(img_gray), img_gray.shape)
    else:
        ys, xs = np.where(img_th1)
        if xs.size == 0 or ys.size == 0:
            using_img_th1 = False
            seed = np.unravel_index(np.argmin(img_gray), img_gray.shape)
        else:
            center_y = int(np.mean(ys))
            center_x = int(np.mean(xs))
            seed = (center_y, center_x)

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

            if using_img_th1 == False:
                if (0 <= nx <= h-1) and (0 <= ny <= w-1) and (mask[nx, ny] == 0):
                    pixel_val = float(img_gray[nx, ny])

                    if abs(pixel_val - float(img_gray[seed])) <= threshold:
                        mask[nx, ny] = True
                        stack.append((nx, ny))
                        region_pixels.append((nx, ny))

                        region_mean = (region_mean * count + pixel_val) / (count + 1)
                        count += 1                

            else:
                if (0 <= nx < h) and (0 <= ny < w) and not mask[nx, ny] and img_th1[nx, ny]:
                    pixel_val = float(img_gray[nx, ny])

                    if abs(pixel_val - float(img_gray[seed])) <= threshold:
                        mask[nx, ny] = True
                        stack.append((nx, ny))
                        region_pixels.append((nx, ny))

                        count += 1

    if mask.sum() < 5:
        if img_th1 is not None:
            return img_th1
    else:
        return mask