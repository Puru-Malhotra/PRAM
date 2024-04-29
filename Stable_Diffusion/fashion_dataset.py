import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        with open('../Dataset/instructions.json', 'rt') as f:
            self.data = json.load(f)

        # print(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        filename = item['imageFileName']
        prompt = ''
        for desc in item['descriptions']:
            if len(prompt) < len(desc):
                prompt = desc

        source = cv2.imread('../Dataset/sketches/' + filename)
        target = cv2.imread('../Dataset/original_images/' + filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, filename=filename)

