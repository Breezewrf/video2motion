# Pipeline

`source .venv/bin/activate`

## Kimodo branch

`python3 motions_to_smpl/kimodo_converter.py --input ./motion_data/kimodo_npz/run_front.npz --output ./motion_data/smpl_npz/run_front.npz`

`python3 gmr/scripts/smplx_to_robot.py --smplx_file ./motion_data/smpl_npz/run_front.npz --robot unitree_g1_23dof --save_path ./motion_data/gmr_pkl/run_front.pkl`

## GVHMR branch
`cd motions_to_smpl/gvhmr/`
`python3 run.py --video=example/tennis.mp4 -s`

`cd ../..`
`python3 gmr/scripts/gvhmr_to_robot.py --save_path ./motion_data/gmr_pkl/tennis.pkl`

## GMR pkl to csv

`python3 gmr/scripts/batch_gmr_pkl_to_csv.py --folder ./motion_data/gmr_pkl`

`python3 vis_robot_motion.py --robot unitree_g1_23dof --robot_motion_path motion_data/gmr_pkl/run_front.pkl --xyzw`

## Mjlab Train
`cd unitree_rl_mjlab`

`python3 ./scripts/csv_to_npz.py --input-file ../motion_data/gmr_pkl/csv/run_front.csv --output-name run_front.npz --input-fps 30 --output-fps 50 --robot g1_23dof`

`python ./scripts/train.py Unitree-G1-23Dof-Tracking-No-State-Estimation --motion_file=./src/assets/motions/g1_23dof/run_front.npz --env.scene.num-envs=4096`