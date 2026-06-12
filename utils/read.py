import argparse
import numpy as np
import json
import os
import torch

def read_npz_data(file_path, show_first=50, list_only=False):
    try:
        data = np.load(file_path)
        print(f"\nNPZ file: {file_path}")
        print("Available arrays:", data.files)
        if list_only:
            return data
        for key in data.files:
            array = data[key]
            print(f"\nArray '{key}': shape={array.shape}, dtype={array.dtype}")
            flat = array.flatten()
            print(f"First {min(show_first, flat.size)} elements: {flat[:show_first]}")
        return data
    except Exception as e:
        print(f"Error reading NPZ file '{file_path}': {e}")
        return None

def read_metadata(file_path):
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata file: {file_path}")
        print(json.dumps(metadata, indent=2))
        return metadata
    except Exception as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None

def read_pkldata(file_path):
    try:
        import pickle
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            import joblib
            data = joblib.load(file_path)
        print(f"\nPKL file: {file_path}")
        if hasattr(data, "keys"):
            print("Keys:", list(data.keys()))
            for key in data.keys():
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"\nKey '{key}': shape={value.shape}, dtype={value.dtype}")
                    print(f"First 30 elements: {value[:30]}")
                elif isinstance(value, torch.Tensor):
                    print(f"\nKey '{key}': Tensor shape={value.shape}, dtype={value.dtype}")
                    print(f"First 30 elements: {value.flatten()[:30]}")
                else:
                    print(f"\nKey '{key}': data: {value}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
        else:
            print("Type:", type(data))
            print("Data:", data)
        return data
    except Exception as e:
        print(f"Error reading PKL file '{file_path}': {e}")
        return None

def read_bvh_data(file_path):
    """
    Parse BVH (Biovision Hierarchy) motion capture file.
    
    Returns a dictionary with:
    - 'hierarchy': list of joint definitions with offsets
    - 'motion_data': numpy array of shape (num_frames, num_channels)
    - 'frame_time': time per frame in seconds
    """
    try:
        hierarchy = []
        motion_data = []
        frame_time = 0.0
        in_motion = False
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"\nBVH file: {file_path}")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("HIERARCHY"):
                in_motion = False
                continue
            elif line.startswith("ROOT") or line.startswith("JOINT"):
                joint_type = line.split()[0]
                joint_name = line.split()[1]
                hierarchy.append({"type": joint_type, "name": joint_name})
            elif line.startswith("MOTION"):
                in_motion = True
                continue
            elif line.startswith("Frames:"):
                num_frames = int(line.split(":")[1].strip())
                print(f"Number of frames: {num_frames}")
            elif line.startswith("Frame Time:"):
                frame_time = float(line.split(":")[1].strip())
                print(f"Frame time: {frame_time}")
            elif in_motion and line and not line.startswith("#"):
                # Parse motion data (frame values)
                try:
                    values = [float(x) for x in line.split()]
                    motion_data.append(values)
                except ValueError:
                    # Skip lines that can't be converted to floats
                    pass
        
        motion_array = np.array(motion_data, dtype=np.float32)
        print(f"Motion data shape: {motion_array.shape}")
        print(f"Detected {len(hierarchy)} joints")
        
        if motion_array.size > 0:
            print(f"\nFirst frame motion data:")
            print(motion_array[0])
        
        result = {
            'hierarchy': hierarchy,
            'motion_data': motion_array,
            'frame_time': frame_time
        }
        
        return result
    except Exception as e:
        print(f"Error reading BVH file '{file_path}': {e}")
        return None

def resolve_path(base_path, fn):
    if fn is None:
        return None
    if os.path.isabs(fn):
        return os.path.expanduser(fn)
    return os.path.join(base_path, fn)

if __name__ == "__main__":
    default_base = "/home/breeze/Desktop/workplace/Humanoid/GMR"
    parser = argparse.ArgumentParser(description="Read motion dataset files (npz/json/pkl/bvh).")
    parser.add_argument("--base-path", "-b", default=default_base,
                        help="Base directory for dataset (expanded).")
    parser.add_argument("--npz", "-n", default="motion.npz",
                        help="NPZ filename or absolute path (default: motion.npz in base-path).")
    parser.add_argument("--json", "-j", default=None,
                        help="JSON metadata filename or absolute path (optional).")
    parser.add_argument("--pkl", "-p", default=None,
                        help="PKL filename or absolute path (optional).")
    parser.add_argument("--bvh", "-bvh", default=None,
                        help="BVH motion capture filename or absolute path (optional).")
    parser.add_argument("--show-first", "-s", type=int, default=50,
                        help="Number of elements to show from each array (default: 50).")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list arrays in the NPZ without dumping data.")
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)
    npz_file = resolve_path(base_path, args.npz)
    json_file = resolve_path(base_path, args.json)
    pkl_file = resolve_path(base_path, args.pkl)
    bvh_file = resolve_path(base_path, args.bvh)

    if not os.path.isdir(base_path):
        print(f"Warning: base path does not exist or is not a directory: {base_path}")

    if npz_file and os.path.exists(npz_file):
        read_npz_data(npz_file, show_first=args.show_first, list_only=args.list_only)
    else:
        print(f"NPZ file not found: {npz_file}")

    if json_file:
        if os.path.exists(json_file):
            read_metadata(json_file)
        else:
            print(f"JSON file not found: {json_file}")

    if pkl_file:
        if os.path.exists(pkl_file):
            read_pkldata(pkl_file)
        else:
            print(f"PKL file not found: {pkl_file}")

    if bvh_file:
        if os.path.exists(bvh_file):
            read_bvh_data(bvh_file)
        else:
            print(f"BVH file not found: {bvh_file}")
