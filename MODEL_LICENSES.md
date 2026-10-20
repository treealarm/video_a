# Model licences

The worker's own code is under the GNU Affero General Public License v3 (`LICENSE`). The models it
loads are separate files with licences of their own. They are not in git (`models/` is ignored) but
are baked into the `analytics-worker` image under `/models` and published as
`treealarm/analytics-models`, so whoever distributes either is distributing them too.

| File in `models/` | What it is | Source | Licence |
|---|---|---|---|
| `primary_detector.xml/.bin` | person/vehicle detector, YOLO11n exported to OpenVINO IR | [Ultralytics](https://github.com/ultralytics/ultralytics) `yolo11n.pt` | AGPL-3.0 |
| `plate_detector.xml/.bin` | single-class licence-plate detector, a YOLOv8n fine-tune | `best.pt` from [Arijit1080/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR](https://github.com/Arijit1080/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR) | AGPL-3.0, as a derivative of YOLOv8n; the export's own metadata says so (`plate_detector.metadata.yaml`) and the source repository states no other |
| `face_detector.xml/.bin` | OpenVINO Open Model Zoo `face-detection-0205` | [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | Apache-2.0 |
| `person_embedder.onnx` (+ `.onnx.data`) | body re-identification, OSNet, 512-d | **not recorded** | **unknown — to be established** |

## Why the worker is AGPL

Ultralytics licenses both its code and its weights under AGPL-3.0 and holds that software built
around them is covered as well, unless an Ultralytics Enterprise licence is bought. The project is
open source anyway, so rather than argue where that line runs, the program that loads and runs the
YOLO models carries the same licence they do. AGPL's network clause (section 13) is satisfied by
this repository being public: anyone using a deployment over the network can get the source here.

The rest of TreeAlarm VMS (`treealarm/ta_vms`, `treealarm/roi_transcode`, `treealarm/ta_install`)
stays MIT. It does not contain or load these models; it talks to this worker over gRPC, as a
separate program.

## Replacing a model

Swapping a YOLO model for one under a permissive licence (Open Model Zoo, YOLOX, RT-DETR, D-FINE
are Apache-2.0) changes this table and nothing else. The worker could go back to MIT only once no
AGPL model is distributed with it.

The person embedder's origin must be found before the next published `analytics-models` image: a
re-identification model trained on Market-1501 or MSMT17 may carry a non-commercial dataset
licence even when its code is MIT.
