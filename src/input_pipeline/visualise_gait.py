import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from enumvars import Directory
from collections import defaultdict
import re


# Mediapipe's connections for drawing the skeleton
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def visualise_gait(npz_path: str, video_writer: cv2.VideoWriter):
    """
    Loads a processed pose clip and overlays it on the original video for verification.

    Args:
        npz_path (str): Path to the .npz file containing the processed pose data.
    """
    # 1. Parse the .npz path to find the original video and start frame
    filename = os.path.splitext(os.path.basename(npz_path))[0]
    print(filename)
    parts = filename.split("_")
    print(filename)

    start_frame = int(parts[-1])
    base_name = "_".join(parts[:-3])
    video_path = os.path.join(Directory.VIDEO.value, f"{base_name}.mp4")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # 2. Load the processed data
    with np.load(npz_path) as data:
        print(data.keys())
        pose_seq_normalized = data["arr"]
        hips_offset = data["hips"]

    # 3. De-normalize the pose sequence by adding the hip offset back
    pose_seq = pose_seq_normalized + hips_offset[:, np.newaxis, :]

    # 4. Load video and prepare for playback
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the starting frame of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 5. Loop through the clip frames and draw the skeleton
    for frame_idx in range(len(pose_seq)):
        ok, frame = cap.read()
        if not ok:
            print("out of frames")
            break

        # Get the landmarks for the current frame
        landmarks = pose_seq[frame_idx]

        # Convert normalized landmark coordinates to pixel coordinates
        points = []
        for lm in landmarks:
            x, y = int(lm[0] * width), int(lm[1] * height)
            points.append((x, y))

        # Draw the skeleton connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if points[start_idx] and points[end_idx]:
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        # Draw the landmark points
        for point in points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

        video_writer.write(frame)
        # Allow breaking the loop by pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()


def main():
    """
    Finds all processed .npz files, groups them by their base video name,
    and creates a stitched MP4 of the base video
    """
    npz_files = sorted(glob.glob(os.path.join(Directory.POSEDATANPY.value, "*.npz")))
    if not npz_files:
        print(f"No processed .npz files found in '{Directory.POSEDATANPY.value}'")
        print("Please run `extract_poses.py` first.")
        return

    # Group .npz files by their video base name
    grouped_files = defaultdict(list)
    for npz_path in npz_files:
        filename = os.path.splitext(os.path.basename(npz_path))[0]
        base_name = "_".join(filename.split("_")[:-3])
        if base_name:
            grouped_files[base_name].append(npz_path)

    for base_name, clip_paths in grouped_files.items():
        clip_paths.sort(
            key=lambda p: int(
                re.search(r"__(\d+)(?:\.npz)?$", os.path.basename(p)).group(1)
            )
        )

    if not grouped_files:
        print(
            "Could not find any .npz files with a valid naming convention (e.g., 'basename_tag_tag_frame')."
        )
        return

    print(f"Found {len(npz_files)} clips from {len(grouped_files)} source videos.\n")

    # Process each group of files into its own video, enumerating by base name
    for i, (base_name, clip_paths) in enumerate(grouped_files.items(), 1):
        print(
            f"[{i}/{len(grouped_files)}] Processing {len(clip_paths)} clips for base video: '{base_name}'"
        )

        output_filename = f"{base_name}_gait_visualization.mp4"
        source_video_path = os.path.join(Directory.VIDEO.value, f"{base_name}.mp4")

        if not os.path.exists(source_video_path):
            print(f" Error: Source video not found at '{source_video_path}'. Skipping.")
            continue

        # Determine video properties from the source video
        cap = cv2.VideoCapture(source_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        cap.release()

        # Initialize VideoWriter for this group
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f" Error: Could not create output file '{output_filename}'.")
            continue

        # Process each clip and append to the group's video
        for npz_path in clip_paths:
            visualise_gait(npz_path, video_writer)

        # Finalize the video for this group
        video_writer.release()
        print(f" Saved: '{output_filename}'")

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
