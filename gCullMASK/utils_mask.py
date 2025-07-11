import os
import torch
import numpy as np
import matplotlib.pyplot  as plt

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def setup_mask(data_dir):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Load Grounding DINO from Hugging Face
  model_id = "IDEA-Research/grounding-dino-base"
  processor = AutoProcessor.from_pretrained(model_id)
  dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda").eval()
  root = data_dir.parent.parent.parent / "models"
  root.mkdir(parents=True, exist_ok=True)
  sam2_checkpoint = root / "sam2.1_hiera_large.pt"
  model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
  sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
  predictor = SAM2ImagePredictor(sam2_model)
  return predictor, processor, dino

def get_bounding_boxes(image_pil, prompt, processor, dino):
  inputs = processor(images=image_pil, text=[prompt], return_tensors="pt").to("cuda")
  with torch.no_grad():
    outputs = dino(**inputs)
  results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.3,
    text_threshold=0.25,
    target_sizes=[image_pil.size[::-1]]
  )
  return results[0]["boxes"].cpu().numpy().astype(int)

def get_masks(image_pil, boxes, predictor):
  arr = np.array(image_pil)
  predictor.set_image(arr)
  for box in boxes:
      masks, scores, _ = predictor.predict(box=box, multimask_output=True)
      best_mask = masks[np.argmax(scores)]
  return best_mask

def disp_mask(image_rgb, binary_mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Segmented Car (DINO + SAM)")
    plt.axis("off")
    plt.show()

def sort_images(data_dir):
   root = data_dir #"/content/drive/My Drive/discord_car/"
   image_paths = sorted([
       os.path.join(root, f)
       for f in os.listdir(root)
       if f.lower().endswith(('.jpg', '.jpeg', '.png'))
   ])
   return image_paths