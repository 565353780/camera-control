import torch
import numpy as np

from typing import List, Tuple


def getCroppedImage(
    image: torch.Tensor,
    mask: torch.Tensor,
    safe_pixel_num: int=10,
) -> torch.Tensor:
    '''
    input:
        image: HxWx3, RGB order, 0-1 float32
        mask: HxW, bool/0-1, foreground region
        safe_pixel_num: int, safety padding pixels around mask bbox
    output:
        cropped_image: H'xW'x3, RGB order, 0-1 float32, cropped to mask bbox with safe padding
    '''
    assert image.dim() == 3 and image.shape[2] == 3, \
        f'[ERROR][image::getCroppedImage] image must be HxWx3, got shape {tuple(image.shape)}'
    assert mask.dim() == 2, \
        f'[ERROR][image::getCroppedImage] mask must be HxW, got shape {tuple(mask.shape)}'
    assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1], \
        f'[ERROR][image::getCroppedImage] image and mask spatial size mismatch: ' \
        f'image {tuple(image.shape[:2])} vs mask {tuple(mask.shape)}'
    assert safe_pixel_num >= 0, \
        f'[ERROR][image::getCroppedImage] safe_pixel_num must be non-negative, got {safe_pixel_num}'

    h, w = int(image.shape[0]), int(image.shape[1])

    bool_mask = mask.bool() if mask.dtype != torch.bool else mask

    if not bool_mask.any():
        print('[WARN][image::getCroppedImage] mask is empty, return original image')
        return image.contiguous()

    rows = bool_mask.any(dim=1)
    cols = bool_mask.any(dim=0)

    y_indices = torch.nonzero(rows, as_tuple=False).squeeze(1)
    x_indices = torch.nonzero(cols, as_tuple=False).squeeze(1)

    y_min = int(y_indices.min().item())
    y_max = int(y_indices.max().item())
    x_min = int(x_indices.min().item())
    x_max = int(x_indices.max().item())

    y1 = max(0, y_min - safe_pixel_num)
    x1 = max(0, x_min - safe_pixel_num)
    y2 = min(h, y_max + 1 + safe_pixel_num)
    x2 = min(w, x_max + 1 + safe_pixel_num)

    cropped_image = image[y1:y2, x1:x2, :].contiguous()

    return cropped_image

def getForegroundImage(
    image: torch.Tensor,
    background_color: List[int]=[255, 255, 255],
    bg_color_diff_max: int=4,
    safe_pixel_num: int=10,
) -> torch.Tensor:
    '''
    input:
        image: HxWx3, RGB order, 0-1 float32
        background_color: [R, G, B], 0-255 int
        bg_color_diff_max: int, 0-255, max per-channel diff (in 0-255 scale) to be treated as background
        safe_pixel_num: int, padding pixels around foreground bbox
    output:
        out_image: H'xW'x3, RGB order, 0-1 float32, foreground bbox placed at center of a
                   background-colored canvas expanded by safe_pixel_num on each side
    '''
    assert image.dim() == 3 and image.shape[2] == 3, \
        f'[ERROR][image::getForegroundImage] image must be HxWx3, got shape {tuple(image.shape)}'
    assert len(background_color) == 3, \
        f'[ERROR][image::getForegroundImage] background_color must have 3 channels, got {len(background_color)}'
    assert bg_color_diff_max >= 0, \
        f'[ERROR][image::getForegroundImage] bg_color_diff_max must be non-negative, got {bg_color_diff_max}'
    assert safe_pixel_num >= 0, \
        f'[ERROR][image::getForegroundImage] safe_pixel_num must be non-negative, got {safe_pixel_num}'

    h, w = int(image.shape[0]), int(image.shape[1])
    dtype = image.dtype
    device = image.device

    bg = torch.tensor(background_color, dtype=dtype, device=device) / 255.0
    bg = bg.view(1, 1, 3)

    diff_max = (image - bg).abs().max(dim=2).values
    diff_threshold = float(bg_color_diff_max) / 255.0
    fg_mask = diff_max > diff_threshold

    if not fg_mask.any():
        print('[WARN][image::getForegroundImage] foreground mask is empty, return a background-only image')
        out_image = bg.expand(safe_pixel_num, safe_pixel_num, 3).contiguous()
        return out_image

    rows = fg_mask.any(dim=1)
    cols = fg_mask.any(dim=0)

    y_indices = torch.nonzero(rows, as_tuple=False).squeeze(1)
    x_indices = torch.nonzero(cols, as_tuple=False).squeeze(1)

    y_min = int(y_indices.min().item())
    y_max = int(y_indices.max().item())
    x_min = int(x_indices.min().item())
    x_max = int(x_indices.max().item())

    crop = image[y_min:y_max + 1, x_min:x_max + 1, :]
    crop_h, crop_w = crop.shape[0], crop.shape[1]

    out_h = crop_h + 2 * safe_pixel_num
    out_w = crop_w + 2 * safe_pixel_num

    out_image = bg.expand(out_h, out_w, 3).contiguous().clone()
    out_image[safe_pixel_num:safe_pixel_num + crop_h, safe_pixel_num:safe_pixel_num + crop_w, :] = crop

    return out_image

def getPaddingImages(
    image_list: List[torch.Tensor],
    target_width: int=1024,
    target_height: int=1024,
    background_color: List[int]=[255, 255, 255],
) -> Tuple[torch.Tensor, np.ndarray]:
    '''
    input:
        image_list: [H_ixW_ix3], RGB order, 0-1 float32
    output:
        padding_images: BxHxWx3, RGB order, 0-1 float32
        padding_info: Bx6, int, [x1, y1, x2, y2, w, h], image bbox in padding images, and image sizes
    '''
    assert target_width > 0 and target_height > 0, \
        '[ERROR][image::getPaddingImages] target_width/target_height must be positive'

    batch_size = len(image_list)

    if batch_size == 0:
        padding_images = torch.zeros(
            (0, target_height, target_width, 3),
            dtype=torch.float32,
        )
        padding_info = np.zeros((0, 6), dtype=np.int64)
        return padding_images, padding_info

    ref_image = image_list[0]
    dtype = ref_image.dtype
    device = ref_image.device

    bg = torch.tensor(background_color, dtype=dtype, device=device) / 255.0
    bg = bg.view(1, 1, 3)

    padding_images = bg.expand(target_height, target_width, 3).unsqueeze(0).repeat(
        batch_size, 1, 1, 1
    ).contiguous()

    padding_info = np.zeros((batch_size, 6), dtype=np.int64)

    for i, image in enumerate(image_list):
        assert image.dim() == 3 and image.shape[2] == 3, \
            f'[ERROR][image::getPaddingImages] image[{i}] must be HxWx3, got shape {tuple(image.shape)}'

        h, w = int(image.shape[0]), int(image.shape[1])
        assert h > 0 and w > 0, \
            f'[ERROR][image::getPaddingImages] image[{i}] has invalid size {w}x{h}'

        scale = min(target_width / w, target_height / h)
        new_w = max(1, min(target_width, int(round(w * scale))))
        new_h = max(1, min(target_height, int(round(h * scale))))

        x1 = (target_width - new_w) // 2
        y1 = (target_height - new_h) // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        if new_w == w and new_h == h:
            resized = image.to(dtype=dtype, device=device)
        else:
            resized = torch.nn.functional.interpolate(
                image.to(dtype=dtype, device=device).permute(2, 0, 1).unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        padding_images[i, y1:y2, x1:x2, :] = resized
        padding_info[i] = np.array([x1, y1, x2, y2, w, h], dtype=np.int64)

    return padding_images, padding_info

def getUnPaddingImageList(
    padding_images: torch.Tensor,
    padding_info: np.ndarray,
) -> List[torch.Tensor]:
    '''
    input:
        padding_images: BxHxWx3, RGB order, 0-1 float32
        padding_info: Bx6, int, [x1, y1, x2, y2, w, h], image bbox in padding images, and image sizes
    output:
        image_list: [H_ixW_ix3], RGB order, 0-1 float32
    '''
    assert padding_images.dim() == 4 and padding_images.shape[-1] == 3, \
        f'[ERROR][image::getUnPaddingImageList] padding_images must be BxHxWx3, got shape {tuple(padding_images.shape)}'
    assert padding_info.ndim == 2 and padding_info.shape[1] == 6, \
        f'[ERROR][image::getUnPaddingImageList] padding_info must be Bx6, got shape {padding_info.shape}'
    assert padding_images.shape[0] == padding_info.shape[0], \
        '[ERROR][image::getUnPaddingImageList] batch size mismatch between padding_images and padding_info'

    image_list: List[torch.Tensor] = []

    for i in range(padding_images.shape[0]):
        x1, y1, x2, y2, w, h = [int(v) for v in padding_info[i].tolist()]

        assert 0 <= x1 < x2 <= padding_images.shape[2], \
            f'[ERROR][image::getUnPaddingImageList] invalid x range [{x1}, {x2}) for width {padding_images.shape[2]}'
        assert 0 <= y1 < y2 <= padding_images.shape[1], \
            f'[ERROR][image::getUnPaddingImageList] invalid y range [{y1}, {y2}) for height {padding_images.shape[1]}'
        assert w > 0 and h > 0, \
            f'[ERROR][image::getUnPaddingImageList] invalid original size {w}x{h}'

        cropped = padding_images[i, y1:y2, x1:x2, :]

        new_h, new_w = cropped.shape[0], cropped.shape[1]
        if new_w == w and new_h == h:
            image = cropped.contiguous()
        else:
            image = torch.nn.functional.interpolate(
                cropped.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0).contiguous()

        image_list.append(image)

    return image_list
