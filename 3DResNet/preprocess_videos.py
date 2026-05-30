import os
import cv2
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def extract_video_frames(args):
    """
    Extract all frames from an AVI video and save them sequentially as JPEG image files. 

    Args:
        video_path (str): input AVI video path
        output_root_dir (str): root directory for saving frame images

    Returns:
        True: video successfully read and saved
        False: video failed to open or process.
    """
    video_path, output_root_dir = args
    
    # extract class names and video names
    parts = video_path.split(os.sep)
    class_name = parts[-2]
    video_name = os.path.splitext(parts[-1])[0]
    
    # create target save directory
    output_dir = os.path.join(output_root_dir, class_name, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # read video with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\ncannot open video: {video_path}")
        return False
        
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame image naming：frame_00001.jpg, frame_00002.jpg ...
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        
        # write image
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
        
    cap.release()
    return True

def main():
    # dynamically obtain the project root directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # return to project root directory.
    
    # define input and output paths
    video_root_dir = os.path.join(project_root, 'data', 'mini_UCF')
    output_root_dir = os.path.join(project_root, 'data', 'mini_UCF_frames')
    
    print(f"Scanning video paths...: {video_root_dir}")
    # recursively match all .avi video files
    video_paths = glob.glob(os.path.join(video_root_dir, '**', '*.avi'), recursive=True)
    total_videos = len(video_paths)
    print(f"Scan complete! {total_videos} videos found.")
    
    if total_videos == 0:
        print("No videos found. Please check the data structure located at the path `data/mini_UCF`!")
        return

    # construct multiprocess parameter pairs
    task_args = [(path, output_root_dir) for path in video_paths]
    
    # automatically retrieve the number of CPU cores
    num_workers = max(1, cpu_count() - 2) # reserve 2 cores to prevent the system from freezing
    print(f"Enable multiprocess processing and allocate the number of cores (Workers): {num_workers}")
    
    # use a pool to launch parallel tasks, and integrate tqdm to display a progress bar
    success_count = 0
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(extract_video_frames, task_args), total=total_videos, desc="Frame Extraction Progress"):
            if result:
                success_count += 1
                
    print(f"\nPreprocessing complete! Successfully converted {success_count} / {total_videos} videos.")
    print(f"All frame images have been saved to: {output_root_dir}")

if __name__ == '__main__':
    main()