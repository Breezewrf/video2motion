# UniMotion

An universal toolkit for humanoid motion data visualization, transformation and editing.

## Setup

```bash
source .venv/bin/activate
```

Large body models and checkpoints are intentionally ignored by git. Make sure the required files are available under:

```text
- converters/gvhmr/inputs/checkpoints/<subfolders>
- converters/gvhmr/hmr4d/utils/body_model/*.pt, *.pts
- assets/body_models/<subfolders>
```

Create a Huggingface repo for easier internal usage:
```sh
# Upload
python upload_to_hf.py \
	--repo-id breezewrf/unimotion_ckpt \
	--private \
	--create-repo \
  --token $HF_TOKEN \

# Download
python3 download_from_hf.py \
  --repo-id breezewrf/unimotion_ckpt \
  --token $HF_TOKEN
```

## Recommended Pipeline

`pipeline.py` is the recommended automation entry point. `--source` describes the input type:

```text
video   -> processed by --video-method, currently gvhmr
kimodo  -> converted to SMPL/SMPL-X NPZ, then GMR
amass   -> converted/normalized as SMPL/SMPL-X NPZ, then GMR
```

GVHMR is not a source. It is currently the video processing method, so use `--source video --video-method gvhmr`.

Pipeline outputs are grouped by robot:

```text
motion_data/<robot>/gvhmr_hmr4d_pt/
motion_data/<robot>/smpl_npz/
motion_data/<robot>/gmr_pkl/
motion_data/<robot>/gmr_pkl/csv/
```

For example, `--robot unitree_g1_23dof` writes under:

```text
motion_data/unitree_g1_23dof/
```

### Video to GMR PKL

```bash
.venv/bin/python pipeline.py \
  --source video \
  --video-method gvhmr \
  --input converters/gvhmr/example/tennis.mp4 \
  --name tennis \
  --robot unitree_g1_23dof \
  -s
```

Output:

```text
motion_data/unitree_g1_23dof/gvhmr_hmr4d_pt/tennis/hmr4d_results.pt
motion_data/unitree_g1_23dof/gmr_pkl/tennis.pkl
```

### Kimodo to GMR PKL

```bash
.venv/bin/python pipeline.py \
  --source kimodo \
  --input motion_data/unitree_g1_23dof/kimodo_npz/run_front.npz \
  --name run_front \
  --robot unitree_g1_23dof
```

Outputs:

```text
motion_data/unitree_g1_23dof/smpl_npz/run_front.npz
motion_data/unitree_g1_23dof/gmr_pkl/run_front.pkl
```

### AMASS to GMR PKL

```bash
.venv/bin/python pipeline.py \
  --source amass \
  --input path/to/amass_motion.npz \
  --name motion_name \
  --robot unitree_g1_23dof
```

### [Opt] [Kimodo|Video|AMASS] to GMR PKL to CSV

Add `--to-csv` to any pipeline command. It runs all previous stages required for that source and then writes CSV.

```bash
.venv/bin/python pipeline.py \
  --source kimodo \
  --input motion_data/unitree_g1_23dof/kimodo_npz/run_front.npz \
  --name run_front \
  --robot unitree_g1_23dof \
  --to-csv
```

Output:

```text
motion_data/unitree_g1_23dof/gmr_pkl/csv/run_front.csv
```

### [Opt] [Kimodo|Video|AMASS] to GMR PKL to Mjlab NPZ

Add `--to-mjlab-npz` to any pipeline command. It runs all previous stages required for that source, creates CSV if needed, and then writes the mjlab NPZ.

```bash
.venv/bin/python pipeline.py \
  --source kimodo \
  --input motion_data/unitree_g1_23dof/kimodo_npz/run_front.npz \
  --name run_front \
  --robot unitree_g1_23dof \
  --to-mjlab-npz \
  --mjlab-robot g1_23dof \
  --mjlab-input-fps 30 \
  --mjlab-output-fps 50
```

Output:

```text
unitree_rl_mjlab/src/assets/motions/g1_23dof/run_front.npz
```

### [Opt] Visualize

Use `--visualize` on a normal pipeline command:

```bash
.venv/bin/python pipeline.py \
  --source video \
  --video-method gvhmr \
  --input converters/gvhmr/example/tennis.mp4 \
  --name tennis \
  --robot unitree_g1_23dof \
  --visualize
```

## Manual Single-Step Commands

These commands are useful for debugging individual stages.

### Kimodo to SMPL NPZ

```bash
python3 converters/kimodo_converter.py \
  --input ./motion_data/unitree_g1_23dof/kimodo_npz/run_front.npz \
  --output ./motion_data/unitree_g1_23dof/smpl_npz/run_front.npz
```

### SMPL NPZ to GMR PKL

```bash
python3 gmr/scripts/smplx_to_robot.py \
  --smplx_file ./motion_data/unitree_g1_23dof/smpl_npz/run_front.npz \
  --robot unitree_g1_23dof \
  --save_path ./motion_data/unitree_g1_23dof/gmr_pkl/run_front.pkl
```

### Video to GVHMR Results

```bash
cd converters/gvhmr
python3 run.py \
  --video=example/tennis.mp4 \
  --output_root=../../motion_data/unitree_g1_23dof/gvhmr_hmr4d_pt \
  -s
cd ../..
```

Output:

```text
motion_data/unitree_g1_23dof/gvhmr_hmr4d_pt/tennis/hmr4d_results.pt
```

### GVHMR Results to GMR PKL

```bash
python3 gmr/scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file ./motion_data/unitree_g1_23dof/gvhmr_hmr4d_pt/tennis/hmr4d_results.pt \
  --robot unitree_g1_23dof \
  --save_path ./motion_data/unitree_g1_23dof/gmr_pkl/tennis.pkl
```

### Optional GVHMR Results to SMPL NPZ Export

This is for compatibility/debugging only. GMR should consume `hmr4d_results.pt` directly for the video path.

```bash
python3 converters/gvhmr_converter.py \
  --input ./motion_data/unitree_g1_23dof/gvhmr_hmr4d_pt/tennis/hmr4d_results.pt \
  --output ./motion_data/unitree_g1_23dof/smpl_npz/tennis.npz
```

### GMR PKL Folder to CSV

```bash
python3 gmr/scripts/batch_gmr_pkl_to_csv.py \
  --folder ./motion_data/unitree_g1_23dof/gmr_pkl
```

### Visualize Robot Motion

```bash
python3 vis_robot_motion.py \
  --robot unitree_g1_23dof \
  --robot_motion_path motion_data/unitree_g1_23dof/gmr_pkl/run_front.pkl \
  --xyzw
```
GMR exported pkl has xyzw format rotation, remember to use `--xyzw` when visualization.

### CSV to Mjlab NPZ

```bash
cd unitree_rl_mjlab
python3 ./scripts/csv_to_npz.py \
  --input-file ../motion_data/unitree_g1_23dof/gmr_pkl/csv/run_front.csv \
  --output-name run_front.npz \
  --input-fps 30 \
  --output-fps 50 \
  --robot g1_23dof
cd ..
```

### Mjlab Train

```bash
cd unitree_rl_mjlab
python3 ./scripts/train.py \
  Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion_file=./src/assets/motions/g1_23dof/run_front.npz \
  --env.scene.num-envs=4096
cd ..
```

### CSV to IsaacLab NPZ
```bash
cd whole_body_tracking

python scripts/csv_to_npz.py --input_file ./motion_data_pkl/csv/run_front.csv --input_fps 30 --output_name run_front --dof 23 --output_fps 50
```

### IsaacLab Train
The motion will be uploaded to wandb, use registry_name to get the motion data.
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-23dof-Wo-State-Estimation-v0 --registry_name org_name/wandb-registry-motions/motion_name --headless --logger wandb --log_project_name beyondmimic --run_name run_name --max_iterations 15000
```
Replace motion_name and run_name with run_front for easier definition.