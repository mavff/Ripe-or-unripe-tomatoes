# Dataset provenance and preparation

## AerialYield-T2M used for the current experiment

[AerialYield-T2M on Zenodo](https://zenodo.org/records/22071809) contains 677 greenhouse images from close-range and drone video. Its [README](https://zenodo.org/records/22071809/files/README.md?download=1) documents 14,190 geometry-filtered detection instances and an official block-grouped split of 477 train, 105 validation, and 95 test images. It has six ripeness stages. This project maps `Green` → `unripe`; `Breakers`, `Turning`, `Pink`, and `Light Red` → `semi_ripe`; **only `Red` (>90% red) → `ripe`**. This threshold was chosen by the project owner.

The source YOLO ZIP contains 989 folder entries for only 677 unique image names. In particular, 312 images occur in more than one train/validation/test folder with identical content. Using those folder boundaries directly would leak images into evaluation. `data prepare-aerial` reads the official image memberships from `dataset_detection_coco.zip` and selects only the matching YOLO image and label in each folder. The processed splits are disjoint. The source has 10,050/1,938/2,202 detection labels in train/validation/test, including only 105/21/14 `ripe` fruits. One exact duplicate label in validation and one in test are removed, as the Ultralytics loader also does; the processed counts are 10,050/1,937/2,201. The source's acquisition sessions are not held out, so test results still may not transfer to a different greenhouse.

Download both verified ZIPs and prepare the data with:

```bash
fruit-harvest data fetch-aerial
fruit-harvest data prepare-aerial --ripe-stage red
```

The downloader checks the published MD5 values before extracting. Both raw ZIPs and all prepared images remain outside Git. The dataset's [LICENSE](https://zenodo.org/records/22071809/files/LICENSE?download=1) is CC BY 4.0. Published example crops credit Afeefa Azam, link the source and license, and describe the annotations added here.

## AgRobTomato used for the first executable workflow

[AgRobTomato on Zenodo](https://zenodo.org/records/5596799) contains 449 greenhouse images and Pascal VOC boxes. The published archive is `Dataset-Greenhouse_Tomato_AgRob.zip` (MD5 `890666716924415720f073b06a9a02a3`). Download and extract it under `data/raw/`; the source root must contain `Annotations/`, `JPEGImages/`, and `ImageSets/Main/default.txt`.

The actual archive has 6,084 annotated fruits: 5,594 `unriped`, 276 `breaking`, 184 `reddish`, and 30 `riped`. We map `unriped` → `unripe`, `breaking` and `reddish` → `semi_ripe`, and `riped` → `ripe`. This preserves the agreed decision that only fully ripe fruit may be harvested. The source does not include a train/test split. We group successive frames in blocks of 20, then assign whole blocks to train, validation, or test with seed 42. The resulting split and class counts are recorded in `dataset_manifest.json`.

Only 30 fully ripe objects are available. Consequently, recall and precision for `ripe` may vary strongly with a few detections. Do not infer field performance from this dataset alone. Although the Zenodo page does not visibly display a license, its [official API record](https://zenodo.org/api/records/5596799) specifies `cc-by-4.0`. We publish only two annotated test examples with [credit and modification details](../output/examples/ATTRIBUTION.md), not the full dataset.

## tomatOD adapter

[tomatOD](https://github.com/up2metric/tomatOD) has 277 images with 2,418 expert-annotated fruits in three classes and is licensed CC BY-NC-SA 4.0. The official image and annotation downloads returned HTTP 403 when checked on 2026-09-30; a listed mirror also returned HTTP 404. The COCO adapter remains available for local copies of the dataset. Supply its image directory and separate official train and test annotation JSON files. The test image IDs are preserved; validation is sampled from training images with a fixed seed.

## Generated dataset

`data prepare` writes `images/{train,val,test}/`, matching `labels/{train,val,test}/`, `dataset.yaml`, and `dataset_manifest.json`. Each label is `class_id center_x center_y width height` with coordinates normalized to `[0, 1]`. Class IDs are `0=unripe`, `1=semi_ripe`, `2=ripe`. No raw or processed images enter Git.
