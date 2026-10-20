# Model licences

The worker's own code is under the GNU Affero General Public License v3 (`LICENSE`). The models it
loads are separate files with licences of their own. They are not in git (`models/` is ignored) but
are baked into the `analytics-worker` image under `/models` and published as
`treealarm/analytics-models`, so whoever distributes either is distributing them too.

| File in `models/` | What it is | Source | Licence |
|---|---|---|---|
| `primary_detector.xml/.bin` | person/vehicle detector: YOLO26s (COCO), exported to OpenVINO IR with its one-to-many head (output `[1,84,8400]`, NMS in the worker) | Ultralytics `yolo26s.pt` from [huggingface.co/Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26) at commit `070ac3c`, exported by `scripts/fetch-models.sh` | AGPL-3.0 |
| `plate_detector.onnx` | single-class licence-plate detector: a YOLO26n fine-tune, output `[1,5,8400]` | `alpr_26n.onnx` from [huggingface.co/ishaqinu/yolo26n-license-plate-detection](https://huggingface.co/ishaqinu/yolo26n-license-plate-detection) at commit `b48cca5`, unchanged | AGPL-3.0, as a derivative of YOLO26n. Trained on the [License Plate Recognition dataset, version 11](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11) (Roboflow Universe), CC BY 4.0 as reported by the model's author |
| `face_detector.xml/.bin` | face detector: Open Model Zoo `face-detection-0205` | [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), 2023.0 FP32 IR, unchanged | Apache-2.0 |
| `person_embedder.xml/.bin` | body re-identification: Open Model Zoo `person-reidentification-retail-0277`, 256-d | [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), 2023.0 FP32 IR, unchanged | Apache-2.0. Intel does not say what it was trained on (see below) |

`scripts/fetch-models.sh` rebuilds `models/` from these sources and checks every downloaded file
against the checksum it records, so the contents of a published `analytics-models` image can be
traced to the publisher byte for byte. The primary detector is the one file made here rather than
downloaded: it is exported from the checked `yolo26n.pt` with the Ultralytics and OpenVINO
versions the script pins.

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

## What the models were trained on

A licence covers the weights; the data they were trained on can carry terms of its own.

- **YOLO26s** (primary detector): COCO, whose annotations are CC BY 4.0 and whose images carry
  their Flickr authors' licences.
- **Plate detector**: the Roboflow Universe dataset named above. Its author's `ATTRIBUTION.md`
  records it as CC BY 4.0 when the model was prepared, which asks for attribution -- this file
  and the link above are that attribution. The Roboflow page is the authority on its current
  terms.
- **face-detection-0205** and **person-reidentification-retail-0277**: Intel publishes both under
  Apache-2.0 and does not name the training data. The re-identification model is evaluated on
  Market-1501; public person re-identification datasets (Market-1501, MSMT17, DukeMTMC-reID) are
  released for research use, so a commercial deployment should not assume more about this model's
  data than Intel states.

## Embedding spaces

The worker names the model behind every embedding it sends (`embedding_model` in
`proto/analytics.proto`): vectors of different models are not comparable, and the length alone
does not tell them apart -- the OSNet model this worker shipped before was 512-d, the current one
is 256-d, and Open Model Zoo's lighter `person-reidentification-retail-0288` is 256-d as well.
Replacing the person embedder therefore starts a new space: ta_vms compares vectors only within
one model, so detections recorded under an earlier model stay in the archive but are not found by
appearance search from a probe of the new one.
