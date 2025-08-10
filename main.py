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

# Below has a lot of personal values.
# In the sense of these comments won't
# make sense if you're someone else because
# I have specific pictures that I am
# converting.
FILE = "ai-1280-pixabay_spatial.heic"
#DEPTH_ZERO = 5 #-- f1
#DEPTH_ZERO = 14 #-- f1 5517 depth pro, 7.5 for depth anything
DEPTH_ZERO = 4.2 #-- pixabay
DEPTH_ADJ = 1  # multiplier
MODE = 'vkitti'  # useless if depthpro
SIZE = 'vitl'
FOCAL_LENGTH = None  # f1 regular 8k: 5547, 4k 2274
DEPTHPRO = True  # more accurate for medium size images



DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print("Imports Handled")

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


def predict_focal_length_px(image_path: str, focal_length = None, depth = False) -> float:

    image, _, f_px = depth_pro.load_rgb(image_path)

    if f_px or focal_length and not depth:
        print("Focal Length from EXIF extracted / provided")
        return f_px or focal_length

    model, transform = depth_pro.create_model_and_transforms(device=DEVICE, precision=torch.float16)
    image = transform(image)
    
    model.eval()

    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)

    print("Depth Pro estimates focal length")
    if depth:
        return prediction["focallength_px"].cpu().numpy().item(), prediction["depth"].cpu().numpy()
    else:
        return prediction["focallength_px"].cpu().numpy().item()

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


if __name__ == "__main__":

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
    print(disparity)
    image = np.array(Image.open(FILE))  # open image
    warped_left = warp_image_with_disparity(image, disparity, left_view=True)
    warped_right = warp_image_with_disparity(image, disparity, left_view=False)
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
