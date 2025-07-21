import math
import subprocess
from PIL import Image
from pathlib import Path
from gCullUTILS.rich_utils import CONSOLE, status

MAX_RESOLUTION = 1600  # maximum allowed width or height

def compute_downscale_factor(
    sample_image: Path, max_res: int = MAX_RESOLUTION
) -> int:
    """Return 1 if both dims ≤ max_res, else the smallest power-of-two factor
    so that max(width, height) / factor ≤ max_res."""
    w, h = Image.open(sample_image).size
    max_dim = max(w, h)
    if max_dim <= max_res:
        return 1
    factor = 1
    while max_dim / factor > max_res:
        factor *= 2
    return factor

def downscale_paths(
    paths: list[Path],
    factor: int,
    use_nearest: bool = False,
):
    """Downscale each file in `paths` by `factor` (int > 1) via ffmpeg."""
    if factor <= 1:
        return  # nothing to do
    
    if "images" in paths:
       target = "images"
    else:
       target = "masks"

    with status(msg=f"[bold yellow]Downscaling {target}...", spinner="growVertical"):
      for p in paths:
          # build output path: put under sibling folder named "<origdirname>_<factor>"
          out_dir = p.parent.parent / f"{p.parent.name}_{factor}"
          out_dir.mkdir(parents=True, exist_ok=True)
          out_path = out_dir / p.name

          # get scaled size from first image
          w, h = Image.open(p).size
          w2, h2 = math.floor(w/factor), math.floor(h/factor)

          # build ffmpeg command
          vf = f"scale={w2}:{h2}" + (":flags=neighbor" if use_nearest else "")
          cmd = [
              "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-noautorotate",
              "-i", str(p),
              "-q:v", "2",
              "-vf", vf,
              str(out_path)
          ]
          subprocess.run(cmd, check=True)
    CONSOLE.log(f"[bold green]:tada: Done downscaling {target}.")
  
def get_downscale_dir(root):
  sample = next(root.glob("*.[pj][pn]g"), None)
  if sample is None:
      raise RuntimeError(f"No masks found in {root}.  Please process masks first.")
  downscale_factor = compute_downscale_factor(sample)
  down_dir = root.parent / f"{root.name}_{downscale_factor}"
  if downscale_factor > 1 and down_dir.is_dir():
      CONSOLE.log(f"[bold yellow]Found existing downscaled images in [bold purple]{down_dir}[bold yellow]; using these for faster processing.")
      target_dir = down_dir
  else:
      # either df == 1, or images_{df} doesn't exist → build it
      target_dir = down_dir if downscale_factor > 1 else root
      if downscale_factor > 1:
          CONSOLE.print(f"[bold yellow] Downscaling [bold purple]{root} [bold yellow] → [bold purple] {down_dir}[bold yellow] by factor  of [bold green]{downscale_factor}")
          all_masks = sorted(root.glob("*.[pj][pn]g"))
          downscale_paths(all_masks, downscale_factor, use_nearest=False)
          target_dir = down_dir
  return target_dir, downscale_factor