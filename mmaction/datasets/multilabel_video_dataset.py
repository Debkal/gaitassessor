from mmaction.datasets import VideoDataset
from mmaction.registry import DATASETS
import csv


@DATASETS.register_module()
class MultiLabelVideoDataset(VideoDataset):
    """Video dataset that supports multi-label annotations from CSV"""

    def load_data_list(self):
        """Load annotation file to get video information."""
        exists = False
        data_list = []

        # Read CSV with all columns
        with open(self.ann_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Get video filename from 'id' column
                video_id = row.get("id", "")
                if not video_id:
                    continue

                # Construct full filename
                filename = f"{self.data_prefix['video']}/{video_id}"

                # Store all labels/annotations as a dictionary
                # Remove 'id' and any source tracking columns
                labels = {
                    k: v
                    for k, v in row.items()
                    if k not in ["id", "csv_source", "video_folder"]
                }

                data_info = {
                    "filename": filename,
                    "label": labels,  # All labels as dict
                    "annotations": labels,  # Keep a copy
                }

                data_list.append(data_info)
                exists = True

        if not exists:
            raise FileNotFoundError(f"Annotation file {self.ann_file} not found")

        return data_list
