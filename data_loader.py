import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset


class ProductDataset(Dataset):
    """
    Generic product dataset loader.

    Reads a CSV file containing product metadata,
    optionally downloads associated images, and
    returns structured samples for multimodal models.
    """

    def __init__(self, data_source, image_dir="images", max_samples=None):
        """
        Args:
            data_source: Path or URI to a CSV file
            image_dir: Local directory to cache images
            max_samples: Optional cap on dataset size
        """
        self.df = pd.read_csv(data_source)

        if max_samples is not None:
            self.df = self.df.sample(max_samples).reset_index(drop=True)

        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        title = str(row.get("title", "")) if pd.notna(row.get("title", "")) else ""
        description = (
            str(row.get("description", ""))
            if pd.notna(row.get("description", ""))
            else ""
        )

        image_url = row.get("image_url")
        item_id = row.get("id", idx)

        image = self._load_image(image_url, item_id)

        return {
            "title": title,
            "description": description,
            "image": image,
            "id": item_id,
            "metadata": row.to_dict(),
        }

    def _load_image(self, url, item_id):
        """
        Download and cache an image locally.
        Returns a PIL image or None if unavailable.
        """
        if not url or pd.isna(url):
            return None

        image_path = os.path.join(self.image_dir, f"{item_id}.jpg")

        if not os.path.exists(image_path):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image.save(image_path)
            except Exception:
                return None

        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None
