# Dataset attribution and layout

The images and annotations in this directory are third-party data, shared here under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Credit the original authors when reusing them. The authors do not endorse this project. The source records and license terms are linked below.

| Directory | Original work and credit | Changes in this project |
| --- | --- | --- |
| `raw/aerial/dataset/detection/yolo/` | Afeefa Azam, [AerialYield-T2M](https://zenodo.org/records/22071809), [license](https://zenodo.org/records/22071809/files/LICENSE?download=1) | Original YOLO export extracted without changing images or labels. Its folder split contains duplicate images across partitions. |
| `aerial-processed/` | Same AerialYield-T2M source and license | Reorganized 677 unique images using the official COCO split membership. Mapped six source stages into `unripe`, `semi_ripe`, and `ripe`; only `Red` (>90% red) is `ripe`. Removed two exact duplicate labels. |
| `raw/Dataset-Greenhouse_Tomato_AgRob/` | Sandro Augusto Magalhães, Germano Moreira, Filipe Neves dos Santos, and Mário Cunha, [AgRobTomato](https://zenodo.org/records/5596799), [license in official record](https://zenodo.org/api/records/5596799) | Original JPEG images and Pascal VOC annotations extracted without modification. |
| `agrob-processed/` | Same AgRobTomato source and license | Converted Pascal VOC boxes to YOLO labels and mapped four source stages into three classes. Assigned frame groups to train, validation, and test with seed 42. |

`raw/aerial-coco.zip` and `raw/aerial-splits.zip` are original source archives used to preserve the published split information. The two larger original ZIP archives (`raw/aerial-yolo.zip` and `raw/agrob.zip`) are omitted because each exceeds GitHub's 100 MiB single-file limit; their extracted content is present under `raw/` and the original downloads are on Zenodo. The prepared data is ready for training and evaluation after cloning. Run commands from the repository root because the committed `dataset.yaml` paths are relative to it.

Class IDs in both prepared datasets: `0=unripe`, `1=semi_ripe`, `2=ripe`. See each `dataset_manifest.json` for exact image membership and object counts, and [dataset documentation](../docs/dataset.md) for processing details. The source and prepared copies coexist so the data preparation and split decisions can be audited.
