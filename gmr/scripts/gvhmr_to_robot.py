import argparse
import pathlib
import os
import time

import numpy as np
import sys
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
GMR_ROOT = PROJECT_ROOT / "gmr"
for path in (PROJECT_ROOT, GMR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/lscm/Desktop/workspace/Humanoid/UniMotion/motions_to_smpl/gvhmr/outputs/demo/tennis/hmr4d_results.pt",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1_23dof", "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1_23dof",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = PROJECT_ROOT / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import joblib
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            joblib.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()