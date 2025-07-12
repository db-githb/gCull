import cv2
import os
from tqdm import tqdm

from PIL import Image
import numpy as np
from pathlib import Path
from rich.console import Console

from gCullMASK.utils_mask import setup_mask, get_bounding_boxes, get_masks, disp_mask, sort_images

CONSOLE = Console()

class MaskProcessor:
  def __init__(
    self,
    data_dir: Path,
    prompt: str = "sky",
    inspect: bool = False,
  ):
    self.data_dir = Path(data_dir)
    self.prompt = prompt
    self.inspect = inspect

  def mask_loop(self, image_paths, predictor, processor, dino):
    
    save_dir = self.data_dir.parent / "masks"
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Producing Masks", disable=False):
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        boxes = get_bounding_boxes(image_pil, self.prompt, processor, dino)

        bool_mask = get_masks(image_pil, boxes, predictor).astype(bool)
        inverted_mask = ~bool_mask

        # Mask the image
        segmented = image_rgb.copy()
        segmented[inverted_mask] = 0

        binary_mask = (inverted_mask.astype(np.uint8)) * 255
        stem = os.path.splitext(os.path.basename(img_path))[0]
        number = stem.split('_')[-1]
        mask_path = f'{save_dir}/mask_{number}.png'

        # Show mask
        if self.inspect and idx % 10 == 0:
            disp_mask(image_rgb, binary_mask)

        # Save mask
        cv2.imwrite(mask_path, binary_mask)

    return save_dir

# main/driver function
  def run_mask_processing(self):
    image_paths = sort_images(self.data_dir)
    predictor, processor, dino = setup_mask(self.data_dir)
    save_dir = self.mask_loop(image_paths, predictor, processor, dino)
    mask_dir = Path(save_dir).resolve()
    linked_name = f"[link=file://{mask_dir}]{mask_dir}[/link]"
    CONSOLE.print(f"🎉 Finished! 🎉 \n ✅ Inspect masks: {linked_name}")