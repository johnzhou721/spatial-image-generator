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

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def pil_path_to_cv2(path: str) -> np.ndarray:
    pil_image = Image.open(path)

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image_np = np.array(pil_image)
    opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return opencv_image

def infer_depth_from_image(image_path, dataset='hypersim', checkpoint_dir='checkpoints', encoder='vits'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    max_depth = 20 if dataset == 'hypersim' else 80
    
    model = DepthAnythingV2Metric(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'{checkpoint_dir}/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.to(DEVICE)
    model.eval()

    image = pil_path_to_cv2(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    depth = model.infer_image(image)  # HxW depth map in meters

    print("Depth Anything Finished")
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


def predict_focal_length_px(image_path: str, focal_length = None, depth = False) -> float:

    image, _, f_px = depth_pro.load_rgb(image_path)

    if (f_px or focal_length) and not depth:
        print("Focal Length from EXIF extracted / provided")
        return f_px or focal_length

    model, transform = depth_pro.create_model_and_transforms(device=DEVICE, precision=torch.float16)
    image = transform(image)
    
    model.eval()

    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px or focal_length)

    print("Depth Pro estimates focal length")
    if depth:
        return to_numpy(prediction["focallength_px"]).item(), to_numpy(prediction["depth"])
    else:
        return to_numpy(prediction["focallength_px"]).item()

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

def backward_warp_image(image, disparity, left_view=True, iterations=2):
    """
    Backward warp an image using source-aligned disparity with fixed-point iterations.
    """
    H, W = disparity.shape
    C = image.shape[2]
    shift_sign = 0.5 if left_view else -0.5

    # 1. Create target pixel grid
    x_target, y_target = np.meshgrid(np.arange(W), np.arange(H))
    x_target = x_target.astype(np.float32)
    y_target = y_target.astype(np.float32)

    # 2. Initialize source coordinates
    x_source = x_target.copy()

    # 3. Fixed-point iterations to solve x_source = x_target - shift * d[x_source]
    for _ in range(iterations):
        # Bilinear sample disparity at current source positions
        d_interp = cv2.remap(
            disparity.astype(np.float32),
            x_source,
            y_target,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        # Update source coordinates
        x_source = x_target - shift_sign * d_interp

    # 4. Clip coordinates to valid range
    x_source = np.clip(x_source, 0, W - 1)
    y_source = np.clip(y_target, 0, H - 1)

    # 5. Warp image
    warped_img = np.zeros_like(image)
    for c in range(C):
        warped_img[..., c] = cv2.remap(
            image[..., c],
            x_source,
            y_source,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # 6. Warp disparity to target view
    warped_disparity = cv2.remap(
        disparity.astype(np.float32),
        x_source,
        y_source,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped_img, warped_disparity


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
    predicted_depth = smooth_depth(predicted_depth)
    #print(predicted_depth)
    save_scaled_image(predicted_depth, path.splitext(FILE)[0] + '_depth.png')
    print(predicted_focal_length_px)
    disparity = (predicted_focal_length_px * 0.064) * (1 / predicted_depth - 1 / DEPTH_ZERO) * DEPTH_ADJ
    print(disparity)
    save_disparity_8bit(disparity, path.splitext(FILE)[0] + "_disparity.png")
    image = np.array(Image.open(FILE))  # open image
    warped_left, _ = backward_warp_image(image, disparity, left_view=True)
    warped_right, _ = backward_warp_image(image, disparity, left_view=False)
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
