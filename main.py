"""
mkdir -p checkpoints
curl -L -o checkpoints/depth_anything_v2_metric_hypersim_vitb.pth \
"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true"
curl -L -o checkpoints/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
"""


import pillow_heif
pillow_heif.register_heif_opener()
import cv2
import torch
import numpy as np
from PIL import Image
from depth_anything_v2_metric.dpt import DepthAnythingV2 as DepthAnythingV2Metric
import depth_pro
from os import path
import argparse
import hashlib

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def _cache_path(image_path, cache_dir="checkpoints"):
    h = hashlib.md5(image_path.encode()).hexdigest()
    return path.join(cache_dir, f"depthpro_{h}.npz")
def _depth_cache_path(image_path, dataset, encoder, cache_dir="checkpoints"):
    h = hashlib.md5(f"{image_path}_{dataset}_{encoder}".encode()).hexdigest()
    return path.join(cache_dir, f"depth_anything_{h}.npz")

def pil_path_to_cv2(path: str) -> np.ndarray:
    pil_image = Image.open(path)

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image_np = np.array(pil_image)
    opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return opencv_image

def infer_depth_from_image(image_path, dataset='hypersim', checkpoint_dir='checkpoints', encoder='vits'):

    cache_file = _depth_cache_path(image_path, dataset, encoder)

    # ---- LOAD CACHE ----
    if path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        print("Depth Anything cache hit")
        return data["depth"]

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    max_depth = 20 if dataset == 'hypersim' else 80

    model = DepthAnythingV2Metric(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(
        torch.load(f'{checkpoint_dir}/depth_anything_v2_metric_{dataset}_{encoder}.pth',
        map_location='cpu')
    )
    model.to(DEVICE)
    model.eval()

    image = pil_path_to_cv2(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    depth = model.infer_image(image)

    print("Depth Anything Finished")

    # ---- SAVE CACHE ----
    np.savez_compressed(
        cache_file,
        depth=depth
    )

    return depth


def threshold_to_image(array: np.ndarray, value: float, output_path: str):
    """
    Sets all elements in the array <= value to 255, then saves as an image.

    Parameters:
        array (np.ndarray): Input numeric array.
        value (float): Threshold value.
        output_path (str): File path to save the resulting image.
    """
    # Copy array to avoid changing the original
    result = array.copy()

    # Apply threshold — set values <= value to 255
    result = np.where(result <= value, 255, 0)

    # Convert to uint8 for image representation
    result_uint8 = result.astype(np.uint8)

    # Create Pillow image from array
    img = Image.fromarray(result_uint8)

    # Save image
    img.save(output_path)
    print(f"Thresholds saved to {output_path}")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, (np.ndarray, np.generic)):
        # Already numpy array or numpy scalar
        return x
    else:
        # fallback, e.g. Python scalar
        return np.array(x)


def predict_focal_length_px(image_path: str, focal_length=None, depth=False):

    cache_file = _cache_path(image_path)

    # ---- LOAD CACHE ----
    if depth and path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        print("Depth Pro cache hit")
        return data["focal"].item(), data["depth"]

    image, _, f_px = depth_pro.load_rgb(image_path)

    if (f_px or focal_length) and not depth:
        print("Focal Length from EXIF extracted / provided")
        return f_px or focal_length

    model, transform = depth_pro.create_model_and_transforms(
        device=DEVICE,
        precision=torch.float16
    )
    image = transform(image)
    model.eval()

    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px or focal_length)

    focal = to_numpy(prediction["focallength_px"]).item()
    depth_map = to_numpy(prediction["depth"])

    print("Depth Pro estimates focal length")

    # ---- SAVE CACHE ----
    np.savez_compressed(
        cache_file,
        focal=focal,
        depth=depth_map
    )

    return (focal, depth_map) if depth else focal

import multiprocessing as mp

def warp_row(args):
    y, image_row, disparity_row, W, channels, shift = args
    warped_row = np.full((W, channels), np.nan)
    disparity_buffer = np.full(W, -np.inf)
    for x in range(W):
        d = disparity_row[x]
        target_x = int(round(x + shift * d))
        if 0 <= target_x < W:
            if d > disparity_buffer[target_x]:
                disparity_buffer[target_x] = d
                warped_row[target_x, :] = image_row[x, :]
    return y, warped_row
    
def inpaint_row(args):
    y, row, W, channels, left_view = args
    row = row.copy()
    inpaint = -1 if left_view else 1
    order = range(W) if left_view else range(W-1, -1, -1)
    for x in order:
        if np.isnan(row[x, 0]):
            neighbor_x = x + inpaint
            if 0 <= neighbor_x < W:
                row[x, :] = row[neighbor_x, :]
    inpaint = 1 if left_view else -1
    order = range(W-1, -1, -1) if left_view else range(W)
    for x in order:
        if np.isnan(row[x, 0]):
            neighbor_x = x + inpaint
            if 0 <= neighbor_x < W:
                row[x, :] = row[neighbor_x, :]
                if np.isnan(row[x, 0]):
                    print(f"ASSERT2 {y} {x} {left_view}")
            else:
                row[x, :] = np.zeros(channels, dtype=row.dtype)
    return y, row

def warp_image(image, disparity, left_view=True):
    H, W = disparity.shape
    channels = image.shape[2]
    shift = 0.5 if left_view else -0.5

    args = [(y, image[y], disparity[y], W, image.shape[2], shift) for y in range(H)]

    with mp.Pool(3) as pool:
        results = pool.map(warp_row, args)

    warped_img = np.full((H, W, channels), np.nan)
    for y, warped_row in results:
        warped_img[y] = warped_row

    print("Warp finished (parallel)")
    return warped_img


def inpaint_image(image, warped_img, left_view):
    H, W, channels = warped_img.shape

    args = [(y, warped_img[y], W, image.shape[2], left_view) for y in range(H)]
    with mp.Pool(3) as pool:
        results = pool.map(inpaint_row, args)

    inpainted_img = np.empty_like(warped_img)
    for y, row in results:
        inpainted_img[y] = row

    print("Inpaint finished (parallel)")
    return inpainted_img.astype(image.dtype)


def warp_image_with_disparity(image, disparity, left_view=True):
    warped_img = warp_image(image, disparity, left_view)
    return inpaint_image(image, warped_img, left_view)

def save_scaled_image(arr, path):
    arr = np.asarray(arr, dtype=np.float64)
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(a_max, a_min):
        arr = np.full_like(arr, 127.5)
    else:
        arr = (arr - a_min) / (a_max - a_min) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_disparity_8bit(disparity: np.ndarray, path: str):
    # Function from ChatGPT
    
    # Replace NaNs and infs with zeros
    disp = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0, 255]
    d_min, d_max = disp.min(), disp.max()
    if np.isclose(d_max, d_min):
        disp_norm = np.zeros_like(disp, dtype=np.uint8)
    else:
        disp_norm = ((disp - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    # Save as PNG
    Image.fromarray(disp_norm).save(path)
    print(f"Disparity saved to {path}, min={disp.min():.4f}, max={disp.max():.4f}")

def smooth_depth(depth_map):
    depth_float = depth_map.astype(np.float32)
    
    blurred_depth = cv2.GaussianBlur(depth_float, (3, 3), sigmaX=0, sigmaY=0)
    
    return blurred_depth

import numpy as np
from numba import njit, prange

import numpy as np
from numba import njit, prange

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def gpu_style_fill_numba(image, mask, left_view=True):
    H, W, C = image.shape
    out = image.copy()

    for y in prange(H):
        row = out[y]

        # -------------------------
        # PASS 1
        # -------------------------
        if left_view:
            order_start, order_end, step = 0, W, 1
        else:
            order_start, order_end, step = W - 1, -1, -1

        last_valid = np.full((C,), np.nan, dtype=np.float32)
        pixel_count = 0
        for x in range(order_start, order_end, step):
            if not mask[y, x]:
                pixel_count += 1
                # only accept stable region AFTER run is confirmed
                if pixel_count >= 15:
                    last_valid = row[x].copy()
            else:
                pixel_count = 0
                if not np.isnan(last_valid).all():
                    for c in range(C):
                        row[x, c] = last_valid[c]
                    mask[y, x] = False


    return out

import numpy as np
from scipy.spatial import cKDTree

def warp_image_zbuffer(image, disparity, predicted_depth, left_view=True):
    img = np.asarray(image)
    disp = np.asarray(disparity, dtype=np.float32)
    depth = np.asarray(predicted_depth, dtype=np.float32)

    if img.ndim == 2:
        img = img[..., None]

    h, w, c = img.shape
    img_f = img.astype(np.float32)

    yy, xx = np.indices((h, w), dtype=np.float32)

    # shift direction
    shift = (0.5 if left_view else -0.5) * disp
    x_dst = xx + shift

    x0 = np.rint(x_dst).astype(np.int32)

    # valid pixels
    valid = (x0 >= 0) & (x0 < w)
    valid_flat = valid.ravel()

    yy_i = yy.astype(np.int32)
    xx_i = xx.astype(np.int32)

    depth_flat = depth.ravel()
    img_flat = img_f.reshape(-1, c)

    src_idx = (yy_i * w + xx_i).ravel()[valid_flat]
    dst_idx = (yy_i * w + x0).ravel()[valid_flat]

    d = depth_flat[valid_flat]
    src = img_flat[src_idx]

    # z-buffer resolve
    order = np.lexsort((d, dst_idx))
    dst_idx = dst_idx[order]
    src_idx = src_idx[order]

    first = np.ones(dst_idx.shape[0], dtype=bool)
    first[1:] = dst_idx[1:] != dst_idx[:-1]

    dst_keep = dst_idx[first]
    src_keep = src_idx[first]

    # warped output
    out = np.full((h * w, c), np.nan, dtype=np.float32)
    out[dst_keep] = img_f.reshape(-1, c)[src_keep]

    out = out.reshape(h, w, c)

    # =========================================================
    # 🧠 DEPTH-AWARE HOLE FILLING
    # =========================================================

    mask = np.isnan(out[..., 0])
    out = gpu_style_fill_numba(out, mask, left_view)
    out = gpu_style_fill_numba(out, mask, not left_view)
    

    return out.astype(image.dtype, copy=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spatial images from a single view using Depth Anything V2 or Depth Pro models."
    )
    parser.add_argument(
        '--file', type=str, required=True,
        help="Input image file path"
    )
    parser.add_argument(
        '--depth_zero', type=float, required=True,
        help="Depth of the zero disparity plane"
    )
    parser.add_argument(
        '--depth_adj', type=float, default=1.0,
        help="Multiplier for the final disparity value"
    )
    parser.add_argument(
        '--mode', type=str, choices=['hypersim', 'vkitti'],
        default='hypersim',
        help="Mode setting: 'hypersim' for indoor, 'vkitti' for outdoor scenes (default: %(default)s)"
    )
    parser.add_argument(
        '--size', type=str, default='vitl',
        help="Encoder size (default: %(default)s)"
    )
    parser.add_argument(
        '--focal_length', type=float,
        help="Focal length in pixels; if None, EXIF or Depth Pro estimation is used"
    )
    parser.add_argument(
        '--depthpro', action='store_true',
        help="Use Depth Pro model (default is False)"
    )

    args = parser.parse_args()

    # Assigning args to your original variables:
    FILE = args.file
    print(FILE)
    DEPTH_ZERO = args.depth_zero
    DEPTH_ADJ = args.depth_adj
    MODE = args.mode
    SIZE = args.size
    FOCAL_LENGTH = args.focal_length
    DEPTHPRO = args.depthpro



    if DEPTHPRO:
        predicted_focal_length_px, predicted_depth = predict_focal_length_px(FILE, FOCAL_LENGTH, True)
        threshold_to_image(predicted_depth, value=DEPTH_ZERO, output_path=path.splitext(FILE)[0] + "_threshold.png")
    else:
        predicted_depth = infer_depth_from_image(FILE, MODE, 'checkpoints', SIZE)
        print(predicted_depth.min(), predicted_depth.max())
        threshold_to_image(predicted_depth, value=DEPTH_ZERO, output_path=path.splitext(FILE)[0] + "_threshold.png")
        predicted_focal_length_px = predict_focal_length_px(FILE, FOCAL_LENGTH)
    #print(predicted_depth)
    save_scaled_image(predicted_depth, path.splitext(FILE)[0] + '_depth.png')
    print(predicted_focal_length_px)
    disparity = (predicted_focal_length_px * 0.064) * (1 / predicted_depth - 1 / DEPTH_ZERO) * DEPTH_ADJ
    print(disparity)
    save_disparity_8bit(disparity, path.splitext(FILE)[0] + "_disparity.png")
    image = np.array(Image.open(FILE))  # open image
    warped_left = warp_image_zbuffer(image, disparity, predicted_depth, left_view=True)
    warped_right = warp_image_zbuffer(image, disparity, predicted_depth, left_view=False)
    Image.fromarray(warped_left).save(path.splitext(FILE)[0] + '_left.png')
    Image.fromarray(warped_right).save(path.splitext(FILE)[0] + '_right.png')

    from spatialconverter import SpatialPhotoConverter
    SpatialPhotoConverter(
        path.splitext(FILE)[0] + '_left.png',
        path.splitext(FILE)[0] + '_right.png',
        path.splitext(FILE)[0] + '_spatial.heic',
        baseline_mm=64.00,
        focal_length=predicted_focal_length_px,
        disparity_adjustment=0.0
    ).convert()
