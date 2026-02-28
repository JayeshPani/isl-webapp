# ISL Dataset Guide (Detection vs Classification)

This project supports two dataset/training modes.

## Mode A: YOLO Detection (`MODEL_MODE=detect`)

Use this when your model predicts sign class from bounding boxes.

### Expected layout

```text
data/
  isl_dataset_detect/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

For each `images/<split>/x.jpg`, create `labels/<split>/x.txt`.

### Label format

Each line in YOLO txt label:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0,1]`.

### Unknown / no-sign guidance (detection)

- Prefer **negative frames with no label rows** (empty txt or no objects), rather than forcing an `unknown` bbox class.
- Keep true sign boxes labeled by sign class.
- Negative no-hand/no-sign examples help reduce false positives.

---

## Mode B: YOLO Classification (`MODEL_MODE=direct_cls`)

Use this when your YOLO model is trained as a classifier (folder-per-class dataset).

### Expected layout

```text
data/
  isl_dataset_cls/
    train/
      A/
      B/
      C/
      ...
      space/
      delete/
      unknown/   # optional but recommended for no-sign frames
    val/
      A/
      B/
      C/
      ...
      space/
      delete/
      unknown/
```

### Unknown / no-sign guidance (classification)

- You can include an explicit `unknown` (or `no_sign`) folder.
- Put frames with no clear sign there (rest pose, blur, out-of-frame, non-sign hand states).

### Important compatibility note

`MODEL_MODE=direct_cls` requires a **classification-trained** model.
A detection-trained `.pt` is not a direct replacement for classification inference.

---

## Shared guidance for both modes

### Train/validation/test split

Recommended initial split:
- train: 70%
- val: 20%
- test: 10%

### Data quality checklist

- Multiple signers (hand shapes, sizes, skin tones)
- Varied lighting (bright/dim/indoor/outdoor)
- Varied backgrounds (plain and cluttered)
- Different camera distances and hand scales
- Motion blur and partial occlusion samples

### Class imbalance mitigation

- Collect more data for rare classes
- Oversample minority classes
- Reduce duplication in dominant classes
- Track confusion matrix after validation

### Augmentation suggestions

- brightness/contrast shifts
- small rotation/translation/scale
- moderate blur/noise
- evaluate mosaic/mixup carefully for hand signs

### Dynamic signs (future)

For sentence-level dynamic signs, collect short clips with sequence labels.
Then plug a sequence model into `DynamicSequenceRecognizerHook` in `backend/inference.py`.
