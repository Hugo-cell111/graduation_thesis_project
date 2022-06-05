import cv2
from torch.utils.data import Dataset
from data.preprocess import img_to_tensor, mask_to_tensor

class loadedDataset(Dataset):
    def __init__(self, txt):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs

    def __getitem__(self, index):
        img, mask = self.imgs[index]
        img = cv2.imread(img)
        mask = cv2.imread(mask)
        if len(mask.shape) != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = mask[:,:,0]
        img = img_to_tensor(img)
        mask = mask_to_tensor(mask)
        mask[mask <= 0.5] = 0
        mask[mask >= 0.5] = 1
        return img, mask

    def __len__(self):
        return len(self.imgs)

"""
OpenCV 中用cv::IMREAD_GRAYSCALE与cv::cvtColor转灰度得到灰度图不一致问题，前者可能有偏差


"""