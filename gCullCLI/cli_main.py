import torch
import argparse
from pathlib import Path
from gCullPY.main.gcull_main import DatasetCull
from gCullMASK.mask_main import MaskProcessor

def main():
    parser = argparse.ArgumentParser(prog="gcull", description="gCull image processing tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mask = sub.add_parser("process-masks", help="Generate binary masks for images from user defined prompt")
    p_mask.add_argument("--data-dir", "-d", required=True,
                        help="Path to images directory")
    p_mask.add_argument("--prompt", "-p", default="sky",
                        help="Detection prompt (default: 'sky')")
    p_mask.add_argument("--inspect", "-i", default=False,
                        help="view mask of first image and every 10th image afterawards")
    
    p_cull = sub.add_parser("cull-model", help="Cull Gaussians from 3DGS model using binary masks")
    p_cull.add_argument("--model-path", "-l", required=True,
                        help="path to 3DGS model's yaml configuration file")
    p_cull.add_argument("--mask-dir", "-m", default=None,
                        help="Path to images directory")
    p_cull.add_argument("--output-dir", "-o", default=None,
                        help="Path to output directory")

    args = parser.parse_args()
    if args.command == "process-masks":
        mp = MaskProcessor(args.data_dir, prompt=args.prompt, inspect=args.inspect)
        mp.run_mask_processing()

    if args.command == "cull-model":
        dc = DatasetCull(Path(args.model_path), mask_dir=args.mask_dir, output_dir=args.output_dir)
        dc.run_cull()

with torch.inference_mode():
    if __name__ == "__main__":
        main()