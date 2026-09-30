# Dataset provenance and preparation

## AgRobTomato used for the first executable workflow

[AgRobTomato on Zenodo](https://zenodo.org/records/5596799) contains 449 greenhouse images and Pascal VOC boxes. The published archive is `Dataset-Greenhouse_Tomato_AgRob.zip` (MD5 `890666716924415720f073b06a9a02a3`). Download and extract it under `data/raw/`; the source root must contain `Annotations/`, `JPEGImages/`, and `ImageSets/Main/default.txt`.

The actual archive has 6,084 annotated fruits: 5,594 `unriped`, 276 `breaking`, 184 `reddish`, and 30 `riped`. We map `unriped` → `unripe`, `breaking` and `reddish` → `semi_ripe`, and `riped` → `ripe`. This preserves the agreed decision that only fully ripe fruit may be harvested. The source does not include a train/test split. We group successive frames in blocks of 20, then assign whole blocks to train, validation, or test with seed 42. The resulting split and class counts are recorded in `dataset_manifest.json`.

Only 30 fully ripe objects are available. Consequently, recall and precision for `ripe` may vary strongly with a few detections. Do not infer field performance from this dataset alone. The Zenodo page does not show an explicit license in its metadata; the repository does not redistribute its images.

## tomatOD adapter

[tomatOD](https://github.com/up2metric/tomatOD) has 277 images with 2,418 expert-annotated fruits in three classes and is licensed CC BY-NC-SA 4.0. The official image and annotation downloads returned HTTP 403 when checked on 2026-09-30; a listed mirror also returned HTTP 404. The COCO adapter remains available for local copies of the dataset. Supply its image directory and separate official train and test annotation JSON files. The test image IDs are preserved; validation is sampled from training images with a fixed seed.

## Generated dataset

`data prepare` writes `images/{train,val,test}/`, matching `labels/{train,val,test}/`, `dataset.yaml`, and `dataset_manifest.json`. Each label is `class_id center_x center_y width height` with coordinates normalized to `[0, 1]`. Class IDs are `0=unripe`, `1=semi_ripe`, `2=ripe`. No raw or processed images enter Git.
