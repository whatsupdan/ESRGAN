import numpy as np
import torch
import cv2
from torchvision.utils import make_grid
import math

def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    out: torch.Tensor = image.flip(-3) #https://github.com/pytorch/pytorch/issues/229
    #out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out

def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)

def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out

def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)

# from https://github.com/ata4/esrgan-launcher/blob/master/upscale.py
def esrgan_launcher_split_merge(input_image, upscale_function, scale_factor=4, tile_size=512, tile_padding=0.125):
    width, height, depth = input_image.shape
    output_width = width * scale_factor
    output_height = height * scale_factor
    output_shape = (output_width, output_height, depth)

    # start with black image
    output_image = np.zeros(output_shape, np.uint8)

    tile_padding = math.ceil(tile_size * tile_padding)
    tile_size = math.ceil(tile_size / scale_factor)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)

            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_padding, 0)
            input_end_x_pad = min(input_end_x + tile_padding, width)

            input_start_y_pad = max(input_start_y - tile_padding, 0)
            input_end_y_pad = min(input_end_y + tile_padding, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = input_image[input_start_x_pad:input_end_x_pad, input_start_y_pad:input_end_y_pad]

            # upscale tile
            output_tile = upscale_function(input_tile)

            # output tile area on total image
            output_start_x = input_start_x * scale_factor
            output_end_x = input_end_x * scale_factor

            output_start_y = input_start_y * scale_factor
            output_end_y = input_end_y * scale_factor

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale_factor
            output_end_x_tile = output_start_x_tile + input_tile_width * scale_factor

            output_start_y_tile = (input_start_y - input_start_y_pad) * scale_factor
            output_end_y_tile = output_start_y_tile + input_tile_height * scale_factor

            # put tile into output image
            output_image[output_start_x:output_end_x, output_start_y:output_end_y] = \
                output_tile[output_start_x_tile:output_end_x_tile, output_start_y_tile:output_end_y_tile]

    return output_image