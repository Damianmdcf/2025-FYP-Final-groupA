import numpy as np
import cv2

def get_mask(img_rgb, img_th1=None, threshold=40):
    """
    Takes in the original image and the annotated mask.
    Computes a new mask using region growing only inside of the annotated mask.
    """
    # For about 10% of the images we don't have an annotated mask. Is toggles on and off based on that.
    using_img_th1 = True
    # get the grayscale version of the original image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if img_th1 is None: # if we don't have an annotated image
        using_img_th1 = False
        # find the darkest reagion in the entrie image (starting point for region growing)
        seed = np.unravel_index(np.argmin(img_gray), img_gray.shape)
    else: # if we do have an annotated image
        ys, xs = np.where(img_th1) 
        if xs.size == 0 or ys.size == 0: # fallback if the size of the annotated image is 0
            using_img_th1 = False 
            # find the darkest reagion in the entrie image (starting point for region growing)
            seed = np.unravel_index(np.argmin(img_gray), img_gray.shape)
        else: # if annotated image has a size
            # get the midtpoint of the annotated mask
            center_y = int(np.mean(ys))
            center_x = int(np.mean(xs))
            # start region growing from there
            seed = (center_y, center_x)
    # get the shape of the image
    h, w = img_gray.shape
    # create an image of all zeros
    mask = np.zeros((h, w), dtype=bool)
    region_pixels = []
    # initialize the stack with only the seed
    stack = [seed]
    # the continouelsy updated mean intensity of the images added to the mask
    region_mean = float(img_gray[seed])
    count = 1

    mask[seed] = True
    region_pixels.append(seed)

    # we always check each of the 4 neighbors
    neighbors = [(-1, 0), (1,0), (0,-1), (0,1)]

    # while the stack is not empty, continue region growing
    while len(stack) > 0:
        (x, y) = stack[-1] # get the last pixel added to the stack (current pixel)
        stack.pop() # remove the current pixel from the stack

        # iterate over all neighbors of the current pixel
        for (dx, dy) in neighbors:
            nx = x + dx # calculate neighbor's x-coordinate
            ny = y + dy # calculate neighbor's y-coordinate

            # if we are not using an annotated mask (img_th1)
            if using_img_th1 == False:
                # check if the neighbor is within image bounds and not already in the mask
                if (0 <= nx <= h-1) and (0 <= ny <= w-1) and (mask[nx, ny] == 0):
                    pixel_val = float(img_gray[nx, ny]) # get the grayscale value of the neighbor pixel

                    # check if the neighbor's intensity is close to the seed pixel's intensity
                    if abs(pixel_val - float(img_gray[seed])) <= threshold:
                        mask[nx, ny] = True # add neighbor to the mask
                        stack.append((nx, ny)) # add neighbor to the stack for further processing
                        region_pixels.append((nx, ny)) # add neighbor to the list of region pixels

                        region_mean = (region_mean * count + pixel_val) / (count + 1) # update the mean intensity of the region
                        count += 1 # increment the count of pixels in the region
            
            # if we are using an annotated mask (img_th1)
            else:
                # check if neighbor is within bounds, not in mask, and is part of the annotated mask
                if (0 <= nx < h) and (0 <= ny < w) and not mask[nx, ny] and img_th1[nx, ny]:
                    pixel_val = float(img_gray[nx, ny]) # get the grayscale value of the neighbor pixel

                    # check if the neighbor's intensity is close to the seed pixel's intensity
                    if abs(pixel_val - float(img_gray[seed])) <= threshold:
                        mask[nx, ny] = True # add neighbor to the mask
                        stack.append((nx, ny)) # add neighbor to the stack for further processing
                        region_pixels.append((nx, ny)) # add neighbor to the list of region pixels

                        count += 1 # increment the count of pixels in the region (mean not updated here for this branch)

    if mask.sum() < 5: # fallback if the size of the mask is less than 5 pixels
        if img_th1 is not None:
            return img_th1 # if an annoted image exist we return that instead
    else:
        return mask # return the region growing mask