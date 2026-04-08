# Dataset Creation Plan

## 1. Posture Categories (YOLOv8)
**Classes:**
1. Standing
2. Sitting
3. Sleeping
4. Charging
5. Running
6. Walking
7. Eating
8. Drinking
9. Dusting
10. Mud-bathing
11. Trunk Up
12. Trunk Down
13. Ear Flapping
14. Tail Swing
15. Calf Under Mother
16. Group Formation

**Requirements:**
- **Images**: 200–500 images per class.
- **Format**: JPG/PNG.
- **Annotation**: Bounding Box (YOLO format: `class_id x_center y_center width height`).

## 2. Behaviour Classes (LSTM/Sequence)
**Classes:**
1. Calm
2. Alert
3. Feeding
4. Social Interaction
5. Migrating
6. Aggressive / Charging
7. Distress / Fear
8. Warning display

**Risk Levels:**
- **High**: Aggressive / Charging, Warning display.
- **Medium**: Alert, Distress / Fear, Feeding (near village).
- **Low**: Calm, Social Interaction, Migrating (in wild).

## 3. Dataset Structure

### Folder Layout
```
/dataset
    /posture_images
        /train
            /images
            /labels
        /val
            /images
            /labels
    /behaviour_sequences
        /videos
        /csv_data
            behaviour_dataset.csv
    data.yaml
```

### CSV/Excel Dataset Template
For the Behaviour Model, we extract features from video frames.

| frame_id | elephant_id | posture_class | x1 | y1 | x2 | y2 | move_dx | move_dy | trunk_angle | ear_freq | tail_freq | behaviour_label | conflict_risk |
|----------|-------------|---------------|----|----|----|----|---------|---------|-------------|----------|-----------|-----------------|---------------|
| 001      | 1           | Walking       | 100| 200| 300| 400| 0.5     | 0.1     | 45          | 0.2      | 0.1       | Migrating       | Low           |
| 002      | 1           | Walking       | 105| 201| 305| 401| 0.5     | 0.1     | 42          | 0.2      | 0.1       | Migrating       | Low           |

## 4. Annotation Guidelines
- **Bounding Boxes**: Tight fit around the visible elephant body.
- **Occlusion**: If >50% occluded, mark as `occluded` or skip if unrecognizable.
- **Truncation**: Annotate visible parts even if truncated by image edge.
- **Attributes**:
    - `Trunk Up`: Only if trunk is clearly raised above head level.
    - `Ear Flapping`: Requires video or motion blur evidence, or wide open ears in static image.

## 5. Augmentation Pipeline (Roboflow Settings)
- **Flip**: Horizontal (Yes), Vertical (No).
- **Rotation**: +/- 15 degrees.
- **Crop**: 0% to 20% zoom.
- **Brightness**: +/- 25%.
- **Blur**: Gaussian Blur (up to 1.5px) to simulate motion/focus issues.
- **Noise**: Up to 5% salt-and-pepper noise (low light simulation).

## 6. Data.yaml Template
```yaml
names:
  0: Standing
  1: Sitting
  2: Sleeping
  3: Charging
  4: Running
  5: Walking
  6: Eating
  7: Drinking
  8: Dusting
  9: Mud-bathing
  10: Trunk Up
  11: Trunk Down
  12: Ear Flapping
  13: Tail Swing
  14: Calf Under Mother
  15: Group Formation
nc: 16
train: ./train/images
val: ./val/images
```
