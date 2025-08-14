import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarkerOptions,PoseLandmarker
import csv
import enumvars as ev

video = ev.Directory.VIDEO.value
posenpy = ev.Directory.POSE_DATA_NPY.value
posecsv = ev.Directory.POSE_DATA_CSV.value

clip_frame_len = 64  # chunking clip frames

class Pose_extractor:
    def __init__(self, scheme):
        self.scheme = scheme

    def process_video(video_path: str):
        base_options = BaseOptions(model_asset_path=ev.Directory.POSE_HEAVY_MODEL.value)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=ev.Pose_config.VISION_RUNNING_MODE.value,
            min_pose_detection_confidence=ev.Pose_config.MIN_DETETECTION_CONFIDENCE.value,
            min_tracking_confidence=ev.Pose_config.MIN_TRACKING_CONFIDENCE.value,
            output_segmentation_masks=ev.Pose_config.SEGMENT_MASK.value,
        )

        landmarker = PoseLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []

        for start in range(0, total - 1, clip_frame_len):
            seq = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(clip_frame_len):
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                mp_img = mp.Image(image_format = mp.ImageFormat.SRGB,
                                  data =rgb)

                result = landmarker.detect_for_video(image=mp_img, timestamp_ms=timestamp)

                if result.pose_landmarks:
                    pts = np.array(
                        [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks[0]],
                        dtype=np.float32,
                    )
                else:
                    pts = np.zeros((33, 3), np.float32)
                seq.append(pts)

            if len(seq) == clip_frame_len:
                seq = np.stack(seq)
                hips = (seq[:, 23] + seq[:, 24]) / 2.0
                seq -= hips[:, None, :]

                base = os.path.splitext(os.path.basename(video_path))[0]
                out_npz = f"{base}_clip__{start}.npz"
                np.savez_compressed(os.path.join(posenpy, out_npz), arr=seq, hips=hips)

                label = base.split("_")[0]
                rows.append({"filepath": os.path.join(posenpy, out_npz), "label": label})

        cap.release()
        landmarker.close()
        return rows


def main():
    print(ev.Pose_config.VISION_RUNNING_MODE.value)
    clip_counter = 0
    rows = []
    for vid in tqdm(glob.glob(os.path.join(video, "*.mp4")), desc="Gait videos"):
        rows.extend(Pose_extractor.process_video(vid))
        clip_counter += 1
        print(clip_counter)

    df = pd.DataFrame(rows)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    df.to_csv(os.path.join(posecsv, "all.csv"), index=False)
    train_df.to_csv(os.path.join(posecsv, "train.csv"), index=False)
    val_df.to_csv(os.path.join(posecsv, "val.csv"), index=False)
    print("wrote posecsv to dir")


if __name__ == "__main__":
    main()
