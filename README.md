# TrackGuard-LTE Production

Inference runtime for single-PC and multi-node Jetson deployments. Run all
commands from the repository root using a Python environment suitable for your device.

## YOLO Weights and the VisDrone Dataset

Model weights and datasets are not included in this repository. Train your own
YOLO object detection model using the VisDrone dataset:

- Official dataset: https://github.com/VisDrone/VisDrone-Dataset
- Dataset format and training guide: https://docs.ultralytics.com/datasets/detect/visdrone/
- YOLO implementation: https://github.com/ultralytics/ultralytics

The following training example is a starting point, not a configuration for
reproducing the paper's results. Adjust the architecture, hyperparameters, and
device settings for your application:

```bash
yolo detect train model=yolo11n.pt data=VisDrone.yaml epochs=100 imgsz=640
```

Pass the trained `best.pt` file to `--model`, or copy it to the repository root
as `best_visdrone_full.pt` to match the commands below. Use the appropriate
VisDrone class names. Different weights may produce different detections and
alarms. Follow the dataset terms and the licenses of the implementations you use.

## Environment

Use Python 3.10 and a working inference environment. PyTorch and torchvision
must match the device's CUDA/JetPack version. Do not replace Jetson-specific
wheels with desktop wheels. `requirements-runtime.txt` lists direct dependencies;
it is not a lockfile for the experimental environment. Install FFmpeg when
required by your video source. Run an MQTT broker, such as Mosquitto, separately
on the central machine.

## Single-PC Collision Inference

```bash
python main.py --source "VIDEO_PATH_OR_URL" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --physics --detect collision --no-display --no-output --evidence-dir runtime/single/evidence
```

Remove `--no-display` to show the video. Snapshots are saved on the inference
device under `runtime/single/evidence/red`, `critical`, and `caution`.

## Multi-Node Deployment

1. Run the broker on the central machine on port 1883, accessible to all three Jetson devices.
2. Start the receiver before starting the nodes. Replace `CENTRAL_PC_IP` with the broker's address.

```bash
python multi-nodes/central/main_central.py --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --sync-interval 5 --log-interval 1 --timeout 0 --output-dir runtime/central
```

3. Run one node on each Jetson from the repository root:

```bash
python multi-nodes/jetson/main_trackguard_node.py --source "https://tcnvr3.taichung.gov.tw/b07122e5" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --jetson-id A --camera-id taichung_A --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --device-mode cuda --stage-profile-dir runtime/node_A --evidence-dir runtime/evidence_A --resource-sample-interval 2 --reporting-mode mqtt --episode-gap-seconds 10 --live-refresh-interval 60
```

```bash
python multi-nodes/jetson/main_trackguard_node.py --source "https://tcnvr6.taichung.gov.tw/0a27f44a" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --jetson-id B --camera-id taichung_B --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --device-mode cuda --stage-profile-dir runtime/node_B --evidence-dir runtime/evidence_B --resource-sample-interval 2 --reporting-mode mqtt --episode-gap-seconds 10 --live-refresh-interval 60
```

```bash
python multi-nodes/jetson/main_trackguard_node.py --source "https://tcnvr6.taichung.gov.tw/8582fc8e" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --jetson-id C --camera-id taichung_C --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --device-mode cuda --stage-profile-dir runtime/node_C --evidence-dir runtime/evidence_C --resource-sample-interval 2 --reporting-mode mqtt --episode-gap-seconds 10 --live-refresh-interval 60
```

The URLs above come from the previous experimental setup. Check stream
availability before deployment. For a five-minute test on Linux, prefix the
command with `timeout --signal=INT 300s`; use `86400s` for 24 hours.
The `--mqtt-smoke-test` option tests MQTT messaging only; it does not run
inference or generate snapshots. Verify message reception on the central machine.

Snapshots in `runtime/evidence_A`, `runtime/evidence_B`, and
`runtime/evidence_C` are stored on their respective Jetson devices, with
`red/critical/caution` subfolders. MQTT transmits metadata, not PNG files.
Receiver CSV files are stored on the central machine.

## Package Scope and Validation

This repository contains the inference runtime. Model weights, datasets,
experimental outputs, and archives are excluded. The legacy `main_jetson.py`
entry point, which references unavailable modules, is also excluded. Use
`multi-nodes/jetson/main_trackguard_node.py` for edge inference.

Local checks passed: syntax parsing of 104 Python files, node and central CLI
help, ByteTrack imports, model loading, and CPU prediction on one synthetic
frame using a locally available 10-class model. That model is not distributed
with this repository. These checks do not cover Jetson CUDA inference, live
CCTV streams, or end-to-end MQTT communication. Test on the target devices
before deployment.

## Citation

If you use TrackGuard-LTE in your research, experiments, or publications, please
cite the following paper:

**Anggi Andriyadi, Ding-Hsiang Huang, and Chao-Tung Yang**, "TrackGuard-LTE:
Local Temporal Evidence for Calibration-Light Edge IoT Traffic Collision Alarms
from Distributed CCTV," *IEEE Internet of Things Journal*, 2026.
[https://doi.org/10.1109/JIOT.2026.3745703](https://doi.org/10.1109/JIOT.2026.3745703)

```bibtex
@ARTICLE{11746008,
  author={Andriyadi, Anggi and Huang, Ding-Hsiang and Yang, Chao-Tung},
  journal={IEEE Internet of Things Journal},
  title={TrackGuard-LTE: Local Temporal Evidence for Calibration-Light Edge IoT Traffic Collision Alarms from Distributed CCTV},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Cameras;Videos;Internet of Things;Tracking;Vehicles;Long Term Evolution;Labeling;Trajectory;Contacts;Planing;Edge IoT;traffic collision alarm;CCTV analytics;calibration-light sensing;physics-inspired image-plane vision;multi-object tracking;MQTT-ready event reporting;surrogate safety measures},
  doi={10.1109/JIOT.2026.3745703}}
```
