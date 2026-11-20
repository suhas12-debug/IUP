# LiDAR and Camera Fusion for Reliable Obstacle Detection in Software Defined Vehicles

A sensor-agnostic camera + LiDAR 3D object detector that keeps working when a sensor is missing or degraded, such as a sparse 16-beam LiDAR.

KLE Technological University, Department of Electronics and Communication Engineering (CIM Lab), Independent Research Project (IRP).

## Problem Statement

Software Defined Vehicles depend on perception that stays reliable in real conditions. Most 3D detectors are trained for one fixed sensor setup, usually a dense 64-beam LiDAR, and their accuracy drops sharply when that setup changes. A cheaper 16-beam LiDAR gives far fewer points, a camera can be blocked or poorly lit, and either sensor can fail on the road.

This project builds a single detector that takes a camera, a LiDAR, or both, and detects cars, pedestrians and cyclists as 3D bounding boxes. The goal is that accuracy degrades gracefully when a sensor is removed or the LiDAR is thinned from 64 to 32 to 16 beams. The system is intended for a lab buggy fitted with a Velodyne VLP-16 and a RealSense D435 camera.

## Team Members

| Name | USN |
|---|---|
| Pavankumar Bill | 01FE23BEC156 |
| Suhas Bagewadi | 01FE23BEC202 |
| Varun P | 01FE23BEC201 |

## Technologies and Tools

| Area | Tools |
|---|---|
| Deep learning | Python, PyTorch |
| 3D detection framework | OpenPCDet, PointPillars |
| Camera branch | ResNet-18 image encoder, depth head, Lift-Splat projection to bird's-eye view (BEV) |
| Fusion | Sensor-agnostic attention fusion with modality dropout and LiDAR-to-camera feature distillation |
| Dataset | KITTI 3D Object Detection |
| Training hardware | NVIDIA RTX 2080 Ti workstation, Kaggle GPU notebooks |
| Target hardware | Velodyne VLP-16 LiDAR, Intel RealSense D435 camera, CPU-only PC |
| Deployment (planned) | ONNX export, ROS2 baseline (YOLO + PointPillars) |

## Approach

1. **LiDAR branch:** PointPillars converts the point cloud into a BEV feature map.
2. **Camera branch:** a ResNet-18 encoder predicts per-pixel depth, then each pixel is lifted into 3D and projected onto the same BEV grid. Depth is supervised from LiDAR points projected into the image.
3. **Sensor-agnostic fusion:** the two BEV maps are combined per cell with attention. A learned "absent" embedding stands in for a missing sensor, so one model handles LiDAR-only, camera-only and both.
4. **Detection head:** an anchor-based head predicts 3D boxes from the fused BEV features.
5. **Robust training:** modality dropout randomly hides a sensor during training, beam thinning and height shift imitate a sparse, lower-mounted LiDAR, and a distillation loss teaches the camera path to reproduce LiDAR features.

## Current Progress

**Completed**
- KITTI data and OpenPCDet environment set up and working on the workstation.
- Pretrained PointPillars LiDAR baseline reproduced, and the accuracy drop at 32 and 16 beams measured.
- LiDAR fine-tuning with beam thinning and height shift, which recovered most of the loss at 16 beams.
- Camera branch (ResNet-18, depth head, Lift-Splat) and the sensor-agnostic fusion model implemented.
- A fusion bug found and fixed (the fusion output was ignoring both sensors). Earlier camera and fusion results were discarded and are being re-run.
- LiDAR-to-camera distillation added to improve the camera-only path.

**In progress**
- Fusion training (joint) and a LiDAR-only control run under the same training recipe, being run on Kaggle GPUs.
- Evaluation of camera-only, LiDAR-only and both-sensor modes at 64, 32 and 16 beams.

**Planned**
- Ablations (no modality dropout, no depth loss, no distillation).
- ONNX export and CPU benchmark.
- VLP-16 live data decoder and ROS2 YOLO + PointPillars baseline.
- Camera-LiDAR calibration and mounting height measurement on the buggy, then depth-aware camera input with the D435.

### Baseline results (fine-tuned LiDAR-only PointPillars, KITTI val, 3D AP_R40, Moderate)

| LiDAR beams | Car | Pedestrian | Cyclist |
|---|---|---|---|
| 64 | 75.7 | 49.0 | 62.1 |
| 32 | 70.2 | 46.0 | 52.3 |
| 16 | 59.8 | 39.7 | 37.7 |

Fusion results will be added here once the re-run is complete.

## Repository Structure

This repository holds only the files we wrote or changed. It is not a full copy of OpenPCDet.

```
README.md
docs/
  DETECTION_PLAN.md                    # full project plan
OpenPCDet/                             # files to copy over a clone of OpenPCDet
  pcdet/models/camera_modules/         # camera branch and sensor-agnostic fusion
  pcdet/models/detectors/pointpillar_fusion.py
  pcdet/models/backbones_3d/vfe/pillar_vfe.py
  pcdet/datasets/kitti/kitti_dataset.py            # camera data loading
  pcdet/datasets/kitti/kitti_object_eval_python/   # CPU rotated-IoU evaluation
  tools/cfgs/kitti_models/pointpillar_fusion.yaml
  tools/cfgs/kitti_models/pointpillar_sparse.yaml
irp_results/                           # training and evaluation scripts
  train_sparse.py                      # fine-tuning with beam thinning and height shift
  eval_beams.py                        # evaluation at 64 / 32 / 16 beams
  merge_ckpt.py, final_table.py, diag_camera.py, summarize_results.py, gt_visualize.py
kaggle_run/                            # Kaggle GPU training runner (added separately)
  kernel-metadata.json
  train_kaggle.py
```

## Setup and Usage

1. Clone [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and install it following its instructions.
2. Copy the contents of this repository's `OpenPCDet/` folder over your OpenPCDet clone (same paths), and register `PointPillarFusion` in `pcdet/models/detectors/__init__.py` if it is not already there.
3. Download KITTI 3D Object Detection (velodyne, left color images, calibration, labels) and generate the info files as described in OpenPCDet's `docs/GETTING_STARTED.md`.
4. Fine-tune the LiDAR branch on thinned scans:
   ```bash
   python irp_results/train_sparse.py --cfg_file cfgs/kitti_models/pointpillar_sparse.yaml --batch_size 4 --epochs 20 --extra_tag finetune_sparse
   ```
5. Train the fusion model (modality probabilities are both / LiDAR-only / camera-only):
   ```bash
   LAMBDA_DISTILL=1 LAMBDA_DEPTH=1 FUSION_MODALITY_PROBS="0.4,0.3,0.3" python irp_results/train_sparse.py --cfg_file cfgs/kitti_models/pointpillar_fusion.yaml --batch_size 2 --epochs 20 --extra_tag joint3 --set OPTIMIZATION.LR 0.0003
   ```
6. Evaluate each mode at each beam count (`MODALITY` is `both`, `lidar` or `camera`):
   ```bash
   MODALITY=both python irp_results/eval_beams.py --beams 16 --cfg cfgs/kitti_models/pointpillar_fusion.yaml --ckpt <checkpoint.pth>
   ```

## Acknowledgements

Built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and the KITTI dataset.
