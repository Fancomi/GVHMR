# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates  
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)  
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

## News 🔥

- [2025-03-08] By default not using DPVO. We implemented a SimpleVO, which is more efficient and compatible with GVHMR.
- [2025-03-08] We added a new option `f_mm` to specify the focal length of the fullframe camera in mm.

## Setup

Please see [installation](docs/INSTALL.md) for details.

## Badminton video / LiDAR labels

Use the project-local uv environment described in [INSTALL](docs/INSTALL.md).
The calibrated full-session job fits every athlete, exports neutral SMPL with
24 joints and real fixed24 LiDAR returns, and verifies the paired dataset:

```powershell
uv run --no-sync python tools/run_badminton_pipeline.py --output-root outputs/badminton_corner_fix_v1_continuity
```

Start the review finalizer in a separate process while fitting runs. It waits
for successful full export, then regenerates 10 five-second clips from the
current fits, includes the reported `20261012_163452` scan-9 case, checks both
cameras, evaluates native person masks and court projections, and verifies the
rendered SMPL geometry in an isolated browser:

```powershell
uv run --no-sync python tools/finalize_badminton_reviews.py --output-root outputs/badminton_corner_fix_v1_continuity --wait-for-job
```

`job_status.json` records dataset progress; `review_job_status.json` records the
later review stages and failures. The final review is
`outputs/badminton_corner_fix_v1_continuity/sequence_review/index.html`.
Changing adopted calibration requires a new output root and a complete rerun;
the pipeline rejects mixing frozen calibration versions. Unchanged teacher
caches are reused only after their input/calibration identity is validated.

For the controlled same-cloud, fixed-shape four-athlete comparison, run:

```powershell
uv run --no-sync python tools/compare_badminton_refinement.py --output-root outputs/badminton_corner_fix_v1 --benchmark-prefix outputs/badminton_corner_fix_v1/calibration_refinement/fit_continuity_benchmark.final --variant batch128_weight0.2 --all-tracks --production-root outputs/badminton_corner_fix_v1_continuity --review-dir outputs/badminton_corner_fix_v1/calibration_refinement/all_tracks_comparison
```

The synchronized page separates the fixed-shape experiment from production
shape fitting, retains rejected-frame markers, and displays recorded geometry
without interpolation. Surface residuals, RGB masks and triangulated detector
hips measure agreement with observations; they are not anatomical ground truth.

## Quick Start

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo for GVHMR](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing)

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for GVHMR](https://huggingface.co/spaces/LittleFrog/GVHMR)

### Demo
Demo entries are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO.
We also provide a script `demo_folder.py` to inference a entire folder.
```shell
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### Reproduce
1. **Test**:
To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:
    ```shell
    python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    ```
    To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

2. **Train**:
To train the model, use the following command:
    ```shell
    # The gvhmr_siga24_release.ckpt is trained with 2x4090 for 420 epochs, note that different GPU settings may lead to different results.
    python tools/train.py exp=gvhmr/mixed/mixed
    ```
    During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

# Acknowledgement

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.
