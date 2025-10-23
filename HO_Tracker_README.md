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

## ▶️ Training
<a id="usage"></a>

Design your own training tasks based on the provided examples. The commands below will help you get started:

```bash
### To train the ManipTrans baseline:

# Right-hand only
# Simple retrageting from MANO to dex hand
python scripts/mano2dexhand.py --data_idx e7816@1 --side right --dexhand inspire --headless --iter 7000

# Imitation RL (the residual policy) to track the hand & object trajcetories
python scripts/train.py task=ResDexHand dexhand=inspire side=RH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true rh_base_model_checkpoint=imitator_ckp/imitator_rh_inspire.pth lh_base_model_checkpoint=imitator_ckp/imitator_rh_inspire.pth dataIndices=[e7816@1] actionsMovingAverage=0.4 experiment=baseline 

# to add checkpoint: checkpoint=runs/baseline__10-22-21-02-26/nn/baseline.pth

# Test the trained model
python scripts/train.py task=ResDexHand dexhand=inspire side=RH headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=false rh_base_model_checkpoint=imitator_ckp/imitator_rh_inspire.pth lh_base_model_checkpoint=imitator_ckp/imitator_lh_inspire.pth dataIndices=[e7816@1] actionsMovingAverage=0.4 checkpoint=runs/baseline__10-22-21-02-26/nn/baseline.pth

```

Train your model(s) on the `data/HO-Tracker/data/test_sample` set.


-------------------------------------



> **Note:** Evaluation compliance. Save checkpoints under `runs/` following the naming pattern: `runs/{your exp tag}_{seq id (do not modify)}_{dexhand (i.e. inspire)}_{hand side (e.g. rh, lh, or bih)}__{timestamp}/nn/last_{your exp tag}_ep_{#epoch}_xxxx.pth`.

## ▶️ Evaluation
<a id="eval"></a>

After training, evaluate your model with:

```bash
# To eval the ManipTrans baseline:
python main/rl/eval_rollout.py --tag baseline --dexhand inspire --extra "rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth"
# You can modify the arguments / rollout code according to your needs.
```

For scoring saved rollouts:
```bash
# To eval the scores of the saved rollouts:
python main/rl/eval_score.py
```

You will obtain summary metrics similar to:
```
================ Overall Results ================
Number of successful sequences: X
Average success rate: Single hand:  X, Bi-hand:  X
Average et (cm):  X
Average er (degree):  X
Average ej (cm):  X
Average eft (cm):  X
