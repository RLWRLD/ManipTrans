# HO-Tracker Challenge — HANDS Workshop @ ICCV 2025

https://github.com/KailinLi/HO-Tracker-Baseline/

## Dataset

Sample training data is provided in [`data/HO-Tracker/data/train_sample`](data/HO_Tracker/data/train_sample).

To browse the dataset locally:

```bash
# Step 1: Download the MANO model from https://mano.is.tue.nl/downloads/
#         Place the extracted MANO assets under the `data/` directory
#         (e.g., `data/mano_v1_2`).

# Step 2: Download the sample data from the repo, LiKailin/HO-Tracker @ HF
`python /data/HO-Tracker/download_repo.py`

# Step 3: Launch the viewer
`python vis_ho_tracker_data.py`

* `h1o1`: one hand, one object
* `h2o1`: two hands, one object
* `h2o2`: two hands, two objects
```

> Note (OakInk V2)
> 
> In OakInk V2, MANO parameters are obtained by fitting to SMPL-X meshes. As a result, hand keypoints computed from MANO may differ from the original OakInk V2 keypoints by a few millimeters. Please choose the set that best suits your use case.
