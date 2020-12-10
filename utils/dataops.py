import numpy as np
import torch
import gc

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

def auto_split_upscale(lr_img, upscale_function, scale=4, overlap=32, current_depth=1):
    failed = False
    try:
        result = upscale_function(lr_img)
        return result, current_depth
    except RuntimeError as e:
        # Check to see if its actually the CUDA out of memory error
        if 'allocate' in str(e):
            failed = True
            # Collect garbage
            torch.cuda.empty_cache()
            gc.collect()
        # Re-raise the exception if not. I have no idea if this will ever be called but its here just in case
        else:
            raise RuntimeError(e)
    finally:
        # Collect garbage again just in case
        gc.collect()
        torch.cuda.empty_cache()

    if failed:
        h, w, c = lr_img.shape

        # Split image into 4ths
        top_left = lr_img[:h//2 + overlap, :w//2 + overlap, :]
        top_right = lr_img[:h//2 + overlap, w//2 - overlap:, :]
        bottom_left = lr_img[h//2 - overlap:, :w//2 + overlap, :]
        bottom_right = lr_img[h//2 - overlap:, w//2 - overlap:, :]
        splits = [top_left, top_right, bottom_left, bottom_right]

        # Recursively upscale the quadrants
        # After we go through the top left quadrant, we know the maximum depth and no longer need to test for out-of-memory
        top_left_rlt, depth = auto_split_upscale(top_left, upscale_function, scale=scale, overlap=overlap, current_depth=current_depth+1)
        top_right_rlt = split_known_depth(top_right, upscale_function, scale=scale, overlap=overlap, max_depth=depth, current_depth=current_depth+1)
        bottom_left_rlt = split_known_depth(bottom_left, upscale_function, scale=scale, overlap=overlap, max_depth=depth, current_depth=current_depth+1)
        bottom_right_rlt = split_known_depth(bottom_right, upscale_function, scale=scale, overlap=overlap, max_depth=depth, current_depth=current_depth+1)

        # Define output shape
        out_h = h * scale
        out_w = w * scale
        out_c = c
        out_overlap = overlap * scale

        # Create blank output image
        output_img = np.zeros((out_h, out_w, out_c), np.uint8)

        # Fill output image with tiles
        output_img[:out_h//2 + out_overlap//2, :out_w//2 + out_overlap//2, :] = top_left_rlt[:-out_overlap//2, :-out_overlap//2, :]
        output_img[:out_h//2 + out_overlap//2, out_w//2 - out_overlap//2:, :] = top_right_rlt[:-out_overlap//2, out_overlap//2:, :]
        output_img[out_h//2 - out_overlap//2:, :out_w//2 + out_overlap//2, :] = bottom_left_rlt[out_overlap//2:, :-out_overlap//2, :]
        output_img[out_h//2 - out_overlap//2:, out_w//2 - out_overlap//2:, :] = bottom_right_rlt[out_overlap//2:, out_overlap//2:, :]

        torch.cuda.empty_cache()
        gc.collect()

        return output_img, depth

def split_known_depth(lr_img, upscale_function, scale=4, overlap=32, max_depth=1, current_depth=1):
    if max_depth == current_depth:
        result = upscale_function(lr_img)
        return result
    else:
        h, w, c = lr_img.shape

        # Split image into 4ths
        top_left = lr_img[:h//2 + overlap, :w//2 + overlap, :]
        top_right = lr_img[:h//2 + overlap, w//2 - overlap:, :]
        bottom_left = lr_img[h//2 - overlap:, :w//2 + overlap, :]
        bottom_right = lr_img[h//2 - overlap:, w//2 - overlap:, :]

        # Recursively upscale the quadrants
        top_left_rlt = split_known_depth(top_left, upscale_function, scale=scale, overlap=overlap, max_depth=max_depth, current_depth=current_depth+1)
        top_right_rlt = split_known_depth(top_right, upscale_function, scale=scale, overlap=overlap, max_depth=max_depth, current_depth=current_depth+1)
        bottom_left_rlt = split_known_depth(bottom_left, upscale_function, scale=scale, overlap=overlap, max_depth=max_depth, current_depth=current_depth+1)
        bottom_right_rlt = split_known_depth(bottom_right, upscale_function, scale=scale, overlap=overlap, max_depth=max_depth, current_depth=current_depth+1)

        # Define output shape
        out_h = h * scale
        out_w = w * scale
        out_c = c
        out_overlap = overlap * scale

        # Create blank output image
        output_img = np.zeros((out_h, out_w, out_c), np.uint8)

        # Fill output image with tiles
        output_img[:out_h//2 + out_overlap//2, :out_w//2 + out_overlap//2, :] = top_left_rlt[:-out_overlap//2, :-out_overlap//2, :]
        output_img[:out_h//2 + out_overlap//2, out_w//2 - out_overlap//2:, :] = top_right_rlt[:-out_overlap//2, out_overlap//2:, :]
        output_img[out_h//2 - out_overlap//2:, :out_w//2 + out_overlap//2, :] = bottom_left_rlt[out_overlap//2:, :-out_overlap//2, :]
        output_img[out_h//2 - out_overlap//2:, out_w//2 - out_overlap//2:, :] = bottom_right_rlt[out_overlap//2:, out_overlap//2:, :]

        return output_img
