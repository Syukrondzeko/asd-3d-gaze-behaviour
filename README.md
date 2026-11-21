# asd-3d-gaze-behaviour

Tools for estimating **3D gaze** from video and attaching it to **FLAME / EMOCA face parameters**, so a person's head, expression and eye direction can be analysed and re-rendered together.

This repository is a **modified copy** of the code for the CVPR 2025 paper *"Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels"* by Pierre Vuillecard and Jean-Marc Odobez (Idiap Research Institute). The gaze model, head detector and demo come from that work; see [Credit & license](#credit--license). What this repository adds is in [`flame_gaze/`](flame_gaze/).

| ![Demo 1](./assets/video_output_1.gif) | ![Demo 2](./assets/video_output_2.gif) |
|--------------------------------------|--------------------------------------|

*Demo output of the original gaze model. The colour of the gaze vector runs from blue (looking toward the camera) to green (looking away). Red marks a gaze perpendicular to the camera.*

---

## What this repository adds

| Script | What it does |
|---|---|
| [`flame_gaze/raw_flame_to_npz.py`](flame_gaze/raw_flame_to_npz.py) | Converts EMOCA per-frame output folders (`shape.npy`, `exp.npy`, `pose.npy`, `eye_pose.npy`, …) into one `.npz` with `shape (1,N,100)`, `exp (1,N,50)`, `pose (1,N,6)` and `eye_pose (1,N,6)` |
| [`flame_gaze/prepare_gaze_npz.py`](flame_gaze/prepare_gaze_npz.py) | Reads `gaze_x`, `gaze_y`, `gaze_z` from the gaze CSV written by `demo.py` and stores them as a `gaze` key in the `.npz` |
| [`flame_gaze/render_npz_to_video_with_gaze.py`](flame_gaze/render_npz_to_video_with_gaze.py) | Renders FLAME parameters to video. Gaze vectors are converted to FLAME eye rotations (`eye_pose_params`) and drawn as **arrows** |
| [`flame_gaze/render_npz_to_video_with_eyeball.py`](flame_gaze/render_npz_to_video_with_eyeball.py) | Same as above, but draws the gaze as an **eyeball** overlay |
| [`flame_gaze/gaze_distribution.py`](flame_gaze/gaze_distribution.py) | Computes the gaze distribution and the dominant viewing area from an `.npz` or CSV. Prints a summary and can save JSON, text and a 3×3 heatmap |
| [`flame_gaze/combine_video.py`](flame_gaze/combine_video.py) | Puts two videos side by side |

Each script has `--help`.

### Typical workflow

```bash
# 1. Predict gaze for a video (writes output/<name>_<modality>_predicted_gaze.csv)
python demo.py --input-filename <video.mp4> --output-dir output/ --modality image

# 2. Turn EMOCA per-frame outputs for the same video into one .npz
python flame_gaze/raw_flame_to_npz.py --input <emoca_output>/EMOCA_v2_lr_mse_20 --output sample_input/person.npz

# 3. Add the predicted gaze to the .npz
python flame_gaze/prepare_gaze_npz.py --input sample_input/person.npz \
    --csv output/<name>_image_predicted_gaze.csv --output sample_input/person_with_gaze.npz

# 4a. Analyse where the person looks
python flame_gaze/gaze_distribution.py --input sample_input/person_with_gaze.npz --save-plot gaze.png

# 4b. Re-render the face with gaze (needs the fer-to-feg project, see Setup)
python flame_gaze/render_npz_to_video_with_eyeball.py --input sample_input/person_with_gaze.npz \
    --output render.mp4 --fer-root /path/to/fer-to-feg
```

---

## Setup

```bash
git clone https://github.com/Syukrondzeko/asd-3d-gaze-behaviour.git
cd asd-3d-gaze-behaviour
conda env create -f environment.yaml    # Python 3.10, PyTorch 2.1.1, CUDA 12.1
conda activate gazeCVPR
bash setup.sh                           # downloads the head-detector weights into weights/
```

A pinned `requirements.txt` is also provided.

**Model checkpoints are not included in this repository.** The original repository shipped `checkpoints/gat_stwsge_gaze360_gf.ckpt` and `checkpoints/gat_gaze360.ckpt`. They were removed here, and `checkpoints/` is git-ignored. Get them from the original authors (see the [paper page](https://publications.idiap.ch/publications/show/5585)) and place them in `checkpoints/`.

**Rendering** (`render_npz_to_video_with_*.py`) imports `models/flame.py`, `utils/renderer.py` and `utils/media.py` from a separate **fer-to-feg** project, plus `psbody.mesh`. Point the scripts to that project with `--fer-root /path/to/fer-to-feg` or `export FER_TO_FEG_ROOT=/path/to/fer-to-feg`.

The videos in `data/` are stored with Git LFS. Run `git lfs install && git lfs pull` to download them.

## Gaze demo (from the original repository)

`demo.py` first detects all heads in the input. For each head it then predicts the gaze direction with the Gaze Transformer, draws the result on the image or video, and saves it to the output directory.

```bash
# Image
python demo.py --input-filename data/pexels-jopwell-2422290.jpg --output-dir output/ --modality image

# Video, each frame processed independently
python demo.py --input-filename data/7149282-hd_1280_720_25fps.mp4 --output-dir output/ --modality image

# Video, using temporal information
python demo.py --input-filename data/7149282-hd_1280_720_25fps.mp4 --output-dir output/ --modality video
```

Options:
- `--ckpt-path ./checkpoints/gat_stwsge_gaze360_gf.ckpt` (default): best model, trained on Gaze360 + GazeFollow
- `--ckpt-path ./checkpoints/gat_gaze360.ckpt`: trained on Gaze360 only (the original README calls this flag `--checkpoint`; the code uses `--ckpt-path`)
- `--device cuda`: run on GPU
- `--batch-size` and `--num-workers`: speed up video inference

---

## Credit & license

**Original work:** Pierre Vuillecard and Jean-Marc Odobez, *Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels*, CVPR 2025. [Paper page](https://publications.idiap.ch/publications/show/5585) · [PDF](https://publications.idiap.ch/attachments/papers/2025/Pierre_3DGAZEESTIMATIONINTHEWILD_2025.pdf). Copyright 2023 Idiap Research Institute.

The original code is licensed under **[CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt)** (non-commercial use only). This modified copy is distributed under the same license.

**Changes made in this repository:**
- Added the `flame_gaze/` scripts listed above.
- `demo.py`:
  - fixed `self.self.input_filename` → `self.input_filename` in video reading
  - set the default `--num-workers` to `0`
  - retries with `num_workers=0` when a DataLoader worker fails (for example, low `/dev/shm` in Docker)
- `environment.yaml`: pinned Python 3.10, PyTorch 2.1.1 / torchvision 0.16.1 with CUDA 12.1, numpy 1.26.4, boxmot 10.0.84, av 17.0.1.
- Added `requirements.txt`.
- Removed the bundled model checkpoints; `checkpoints/`, `sample_flame/` and `sample_input/` are git-ignored.

### Third-party resources (from the original repository)

Training data:
- [Gaze360](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md): research use only
- [GazeFollow](http://gazefollow.csail.mit.edu/): non-commercial

Code:
- Built on Swin3D from [omnivore](https://github.com/facebookresearch/omnivore): CC BY-NC 4.0

Demo:
- Face-box tracking: [BoxMOT](https://github.com/mikel-brostrom/boxmot), AGPL-3.0
- Head detector: [yolov5-crowdhuman](https://github.com/MahenderAutonomo/yolov5-crowdhuman), GPL-3.0. It uses [YOLOv5](https://github.com/ultralytics/yolov5) (AGPL-3.0) trained on [CrowdHuman](https://www.crowdhuman.org/) (non-commercial)

## Citation

If you use the gaze model, please cite the original paper:

```bibtex
@INPROCEEDINGS{vuillecard3DGAZEWILD2025,
         author = {Vuillecard, Pierre and Odobez, Jean-Marc},
       projects = {Idiap},
          month = jun,
          title = {Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
           year = {2025},
            pdf = {https://publications.idiap.ch/attachments/papers/2025/Pierre_3DGAZEESTIMATIONINTHEWILD_2025.pdf}
}
```
