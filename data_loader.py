from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
import torch

class RandomCrop(object):

    def __init__(self,output_size):
        self.output_size = output_size
    def __call__(self,img : Image.Image):

        if random.random() >= 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        h, w = img.height, img.width

        top = random.randint(0, h - self.output_size)
        left = random.randint(0, w - self.output_size)

        img = img.crop((left, top, left + self.output_size, top + self.output_size))

        return img

class SRDataset(Dataset):
    def __init__(self, train_list) -> None:
        self.train_list = train_list
        self.transform = transforms.Compose([RandomCrop(1024),transforms.ToTensor()])
    
    def __len__(self):
        # return len(self.train_list)
        return 16
    
    def __getitem__(self, index):
        img = Image.open(self.train_list[0]).convert('RGB')
        hr = self.transform(img)
        lr = F.interpolate(hr.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
        
        return {'hr' : hr, 'lr' : lr}