import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import scipy.fft as fft
import jax
import jax.numpy as jnp
import random
from enum import Enum







__all__ = ["convert_image", "DitherPattern"]
__version__ = "1.0.0"






colorMap = cv2.COLORMAP_JET
#colorMap = cv2.COLORMAP_HOT
#colorMap = cv2.COLORMAP_OCEAN


class DitherPattern(Enum):
    BAYER2X2 = 0,
    BAYER4X4 = 1,
    BAYER8X8 = 2,
    INTERLEAVED_GRADIENT_NOISE = 3,
    BLUE_NOISE = 4





def create_bayer_matrix(n):
    """
    Generates an n x n Bayer matrix.
    n must be a power of 2.
    """
    if n == 1:
        return np.array([[0]])
    else:
        m = n // 2
        b_m = create_bayer_matrix(m)
        
        top_left = 4 * b_m
        top_right = 4 * b_m + 2
        bottom_left = 4 * b_m + 3
        bottom_right = 4 * b_m + 1
        
        return np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])




def create_interleaved_gradient_noise(n):
    img = np.zeros((n, n), dtype=np.float32)

    for x in range(n):
        for y in range(n):
            value = 52.9829189 * (0.06711056 * float(x) + 0.00583715*float(y) % 1.0) % 1.0
            img[x,y] = value

    return img
    img = np.uint8(img * 255.0)
    debug = DEBUG_image(img)
    cv2.imshow("finalHeat",debug)
    img = upscale(img,4)
    cv2.waitKey()
    cv2.imwrite(output,img)
       




def void_and_cluster(size, sigma = 1.9, seed_points_per_dim = -1):
    # the number of seed points = seed_points_per_dim * seed_points_per_dim.
    # by default 1 in 8 points per axis will be a seed point
    if seed_points_per_dim < 0:
        seed_points_per_dim = max(size // 8, 1)

    wrapped_pattern = np.hstack((np.linspace(0, size/2-1, size//2), np.linspace(size/2, 1, size//2)))
    wrapped_pattern = np.exp(-0.5 * wrapped_pattern * wrapped_pattern / (sigma * sigma))
    wrapped_pattern = np.outer(wrapped_pattern, wrapped_pattern)

    wrapped_pattern[0, 0] = np.inf
    lut = jnp.array(wrapped_pattern)
    jax.device_put(lut)

    # tuple pos_xy_source.
    def energy(pos_xy_source):
        return jnp.roll(lut, shift=(pos_xy_source[0], pos_xy_source[1]), axis=(0, 1))

    points_set = []
    bucket_size = size // seed_points_per_dim
    for x in range(0, seed_points_per_dim):
        for y in range(0, seed_points_per_dim):
          points_set.append((random.randint(x * bucket_size, ((x + 1) * bucket_size - 1)),
                         random.randint(y * bucket_size, ((y + 1) * bucket_size - 1))))
    points_set = np.random.permutation(points_set)

    energy_current = jnp.array(sum(energy(p) for p in points_set))
    jax.device_put(energy_current)

    # finds pixel with lowest energy, and updates the energy map to contain it.
    @jax.jit
    def update_step(energy_current):
        pos_flat = energy_current.argmin()
        pos_x, pos_y = pos_flat // size, pos_flat % size
        return energy_current + energy((pos_x, pos_y)), pos_x, pos_y

    final_res = np.zeros_like(lut)
    init_size = seed_points_per_dim * seed_points_per_dim
    for i, p in enumerate(points_set):
        final_res[p[0], p[1]] = i

    for x in range(size * size - init_size):
        energy_current, pos_x, pos_y = update_step(energy_current)
        final_res[pos_x, pos_y] = x + init_size
    return final_res / float(size * size)





def blue_noise(output, size=256):
    noise = void_and_cluster(size) * 255
    noise = upscale(noise,4).astype(np.uint8)
    cv2.imshow("test",noise)
    cv2.imwrite(output,noise)
    cv2.waitKey()
    








def upscale(img,factor):
    return cv2.resize(img,dsize=None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)





def order_pixels(input_image):
     
    h, w = input_image.shape
    flat = input_image.flatten()
    num_pixels = flat.size
    sorted_indices = np.argsort(flat)
    # running argsort again as a reverse lookup
    ranks = np.argsort(sorted_indices)
    normalized_img = ranks.astype(np.float32) / num_pixels
    return normalized_img.reshape(h, w)






def order_pixels_tiled(input_image, tile_size = 16):
    h, w = input_image.shape[:2]
    
    gray = input_image.copy()

    output_img = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # get the window/tile
            tile = gray[y:y+tile_size, x:x+tile_size]
            
            flat_tile = tile.flatten()
            num_pixels = flat_tile.size
            
            # indices of sorted pixels
            sorted_indices = np.argsort(flat_tile)
            
            # running argsort again as a reverse lookup
            ranks = np.argsort(sorted_indices)
            
            normalized_tile = ranks.astype(np.float32) / num_pixels
            reshaped_tile = normalized_tile.reshape(tile.shape)
            output_img[y:y+tile_size, x:x+tile_size] = reshaped_tile

    return output_img









def DEBUG_indexes(indices,shape):
    # create a remapped image of the indices
    factor = shape[0] * shape[1]
    img = indices.reshape(shape)
    img / factor;
    img = cv2.resize(img,dsize=None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(img.astype(np.uint8), colorMap)

def DEBUG_image(img):
    img = upscale(img,8)
    return cv2.applyColorMap(img.astype(np.uint8), colorMap)

def DEBUG_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 5))
    plt.title("Grayscale Image Histogram")
    plt.xlabel("Pixel Value (Intensity)")
    plt.ylabel("Number of Pixels")
    plt.plot(hist)
    # plt.bar(bins[:-1], hist, width=1, color='gray') 
    plt.xlim([0, 256])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()





def round_to_po2(x):
    l = math.log2(x)
    return 2 ** round(l)




def convert_image(file, output=None, resize=None, noiseType=DitherPattern.BAYER4X4, noiseAmount=0.125, tileSize=None, finalUpscale=1, gamma=2.2, repeats=(1,1), debug=False):
    """

    Converts an image to an ordered dither pattern based on luminance.
    The image can be mixed with a dither pattern - like a Bayermatrix - to create an even distribution.
    If the image is not of a power-of-two size it will be rescaled.

    Args:
        file (string or None) :             path to the input image

        output (string) :                   output path - if empty  the file will be called _out and saved next to input

        resize (int,int or None):           optional new size for input image, ie (32,32). Size will be converted to power of two anyway.

        noiseType (DitherPattern) :         a dither pattern to be mixed with the input image

        noiseAmount (float) :               blend amount of the dither pattern

        tilesize (int or None) :            if not None the image will be segmented into tiles before sorting

        finalUpscale (int) :                final scale (default 1)

        gamma (float)       :               input gamma (default 2.2)

        repeats (int,int)   :               optional tiling of the image

        debug (bool)      :                 show intermedieate images


    Returns:

        True on success


    """

    img = cv2.imread(file,0)

    if not resize is None:
        img = cv2.resize(img, resize)

    img = np.float32(img)/255.0
    img = np.power(img,gamma)

    # round to closest power of 2
    new_w  = round_to_po2(img.shape[1])
    new_h  = round_to_po2(img.shape[0])

    new_shape = (new_h, new_w)


    img = cv2.resize(img, (new_w, new_h))
    print(f"new image shape: {img.shape}")

    #smallerAxis = min(new_w,new_h)
    #ssHorizontal = new_w > new_h
    #print("Smaller Axis: " + str(smallerAxis))

    pattern = None




    if noiseType in {DitherPattern.BAYER2X2, DitherPattern.BAYER4X4, DitherPattern.BAYER8X8}:

        bayer = None

        if noiseType == DitherPattern.BAYER2X2:
            bayer = create_bayer_matrix(2)
        elif noiseType == DitherPattern.BAYER4X4:
            bayer = create_bayer_matrix(4)
        else : 
            bayer = create_bayer_matrix(8)
        bayer = np.float32(bayer)/255.0

        print(f"generated bayer pattern: {bayer.shape}")

        factorY = new_shape[0] / bayer.shape[0]
        factorX = new_shape[1] / bayer.shape[1]

        factor = (int(factorY),int(factorX))
        print(f"tiling pattern x*{factor[0]} y*{factor[1]}")
        bayer = np.tile(bayer,factor)
        print(f"tiled bayer pattern: {bayer.shape}")

        pattern = bayer

    elif noiseType == DitherPattern.BLUE_NOISE:
        pattern = void_and_cluster(new_shape[0])

    elif noiseType == DitherPattern.INTERLEAVED_GRADIENT_NOISE:
            pattern = create_interleaved_gradient_noise(new_shape[0])

    else:
        print(f"noiseType {noiseType} not recognized")
        return False



    print(f"pattern shape: {pattern.shape}")
    #composite = cv2.addWeighted(pattern, noiseAmount, img, 1.0-noiseAmount, 0)
    composite = img * (1.0-noiseAmount) + pattern * noiseAmount


    # do the thing
    # either tiled or over the whole image
    if tileSize != None:
        img = order_pixels_tiled(composite, tileSize)
    else:
        img = order_pixels(composite)

    img = img.reshape(new_shape)

    img = np.uint8(img * 255.0)

    #DEBUG_histogram(img)
    debug_pattern   = DEBUG_image(pattern*255)
    debug_composite = DEBUG_image(composite*255)
    debugFinal      = DEBUG_image(img)


    img = np.tile(img,repeats)


    if finalUpscale != 1:
        img = upscale(img, finalUpscale)

    if not output:
        (path, _) = os.path.splitext(file)
        output = path + "_out.png"

    print(f"writing file to {output}\n")
    cv2.imwrite(output,img)

    if debug:
        cv2.imshow("input brightness",debug_original)
        #cv2.imshow("pattern",debug_pattern)
        cv2.imshow("composite",debug_composite)
        cv2.imshow("final",debugFinal)
        cv2.imshow("indices",debug_indices)
        cv2.waitKey()

    return True

