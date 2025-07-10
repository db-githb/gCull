<h1 align="center">gCull</h1>

A command-line tool to clean noisy Gaussian primitives associated with sky and clouds in 3D Gaussian Splatting (3DGS) reconstructions.

<p align="center">
  <img src="images/car_og.png" alt="Original 3DGS Reconstruction" width="45%" />
  <img src="images/car_cull.png" alt="Culled 3DGS Model" width="45%" />
</p>

## 💾 Installation

```bash
conda create -n gcull python=3.11 -c conda-forge
conda activate gcull
cd <project-dir: gCull>
pip install -r requirements.txt
pip install -e .
cd gCullCUDA
pip install -e .
```

## 🎭 Mask Requirements

This tool requires binary mask files for training images:

- One binary mask per training image.

- Sky regions encoded as black (pixel value 0).

 - Non-sky regions encoded as white (pixel value 255).

Place masks in data/<experiment-name>/masks/, ensuring mask filenames match their corresponding image filenames (e.g., frame_0001.png ↔ mask_0001.png).

## 📂 File Structure

The tool requires the following structure:

```text
gCull/
├── data/
│   └── <experiment-name>/
│       ├── colmap/
│       ├── images/
│       ├── masks/
│       └── transforms.json
├── outputs/
│   └── <experiment-name>/
│       └── splatfacto/
│           └── <model-name>/
│               └── config.yml
```

## 🚀 Execution

Run the tool from the project root:

```bash
gcull --load-config <path/to/config.yml>
```

## 📁 Output

The final culled model will be saved under the data folder as:

```
<project-root>/data/<experiment-name>/<experiment-name>_culled.ply
```

