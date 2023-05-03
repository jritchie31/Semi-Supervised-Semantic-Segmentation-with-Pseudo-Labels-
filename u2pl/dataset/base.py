import logging

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, d_list, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        if "Crack_Dataset" in d_list:
            if "unlabeled.txt" in d_list:
                self.list_sample = [
                    [
                        line.strip(),
                        line.strip(),
                    ]
                    for line in open(d_list, "r")
                ]
            elif "val_label.txt" in d_list:
                self.list_sample = [
                    [
                        line.strip(),
                        line.strip()[:44] + "Segmentation" + line.strip()[49:],
                    ]
                    for line in open(d_list, "r")
                ]            
            else:
                self.list_sample = [
                    [
                        line.strip(),
                        line.strip()[:52] + "Segmentation" + line.strip()[-38:],

                    ]
                    for line in open(d_list, "r")
                ]
        else:
            raise "unknown dataset!"

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.usable_sample = len(self.list_sample)
        assert self.usable_sample >= 0
        logger.info("# usable_samples: {}".format(self.usable_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def __len__(self):
        return self.usable_sample
