# Pipeline
`source .venv/bin/activate`

`python3 motions_to_smpl/kimodo_converter.py --input ./motion_data/kimodo_npz/run_front.npz --output ./motion_data/smpl_npz/run_front.npz`

`python3 gmr/scripts/smplx_to_robot.py --smplx_file ./motion_data/smpl_npz/run_front.npz --robot unitree_g1_23dof --save_path ./motion_data/gmr_pkl/run_front.pkl`

`python3 gmr/scripts/batch_gmr_pkl_to_csv.py --folder /home/breeze/Desktop/workplace/Humanoid/UniMotion/motion_data/gmr_pkl`

