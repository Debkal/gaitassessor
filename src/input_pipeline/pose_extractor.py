import numpy as np
import cv2
import os, glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import mediapipe as mp
import os
import csv
from enumvars import Directory as dir

video = dir.VIDEO.value
posenpy = dir.POSEDATANPY.value
posecsv = dir.POSEDATACSV.value

clip_frame_len = 64  # chunking clip frames


class Pose_extractor:
    def __init__(self, scheme):
        self.scheme = scheme

    def process_video(video: str):
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        remainder = total % clip_frame_len

        rows = []

        for start in range(0, total - 1, clip_frame_len):
            seq = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(clip_frame_len):
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_pose.process(rgb)
                if res.pose_landmarks:
                    pts = np.array(
                        [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                else:
                    pts = np.zeros((33, 3), np.float32)
                seq.append(pts)

            if len(seq) == clip_frame_len:
                seq = np.stack(seq)

                hips = (seq[:, 23] + seq[:, 24]) / 2.0
                seq -= hips[:, None, :]

                base = os.path.splitext(os.path.basename(video))[0]
                out_npz = f"{base}_clip__{start}.npz"

                np.savez_compressed(os.path.join(posenpy, out_npz), arr=seq, hips=hips)

                label = base.split("_")[0]
                rows.append(
                    {"filepath": os.path.join(posenpy, out_npz), "label": label}
                )

        cap.release()
        mp_pose.close()
        return rows


def main():
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
