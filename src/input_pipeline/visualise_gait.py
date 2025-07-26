import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from enumvars import Directory


# Mediapipe's connections for drawing the skeleton
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def visualise_gait(npz_path: str):
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

        cv2.imshow(f"Gait Visualization: {os.path.basename(video_path)}", frame)

        # Allow breaking the loop by pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Presents an interactive menu to let the user select a specific clip
    to visualise, or to visualise all clips sequentially.
    """
    # Use sorted() for a consistent order in the menu
    npz_files = sorted(glob.glob(os.path.join(Directory.POSEDATANPY.value, "*.npz")))
    if not npz_files:
        print(f"No processed .npz files found in {Directory.POSEDATANPY.value}")
        print("Please run `extract_poses.py` first.")
        return

    while True:
        print("\nAvailable clips for visualization:")
        for i, f_path in enumerate(npz_files):
            print(f"  [{i + 1}] {os.path.basename(f_path)}")

        print(
            "\nEnter a number to select a clip, 'all' to process every clip, or 'q' to quit."
        )

        choice = input("> ").strip().lower()

        if choice == "q":
            break
        elif choice == "all":
            print(
                "\nVisualizing all clips. Press 'q' in the video window to skip to the next one."
            )
            for npz_path in npz_files:
                print(f"Now showing: {os.path.basename(npz_path)}")
                visualise_gait(npz_path)
            print("\nFinished visualizing all clips.")
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(npz_files):
                    selected_path = npz_files[index]
                    print(f"\nVisualizing clip: {os.path.basename(selected_path)}")
                    print("Press 'q' in the video window to return to the menu.")
                    visualise_gait(selected_path)
                else:
                    print(
                        f"Invalid number. Please enter a number between 1 and {len(npz_files)}."
                    )
            except ValueError:
                print("Invalid input. Please enter a number, 'all', or 'q'.")


if __name__ == "__main__":
    main()
