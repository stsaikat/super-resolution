from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class SRDataset(Dataset):
    def __init__(self, train_list) -> None:
        self.train_list = train_list
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        # return len(self.train_list)
        return 16
    
    def __getitem__(self, index):
        img = Image.open(self.train_list[0]).convert('RGB')
        # print(img.size)
        width, height = img.width, img.height
        width, height = width // 2, height // 2
        # width, height = 512, 512
        hr = img.resize((width*2, height*2), resample=Image.Resampling.NEAREST)
        lr = img.resize((width, height), resample=Image.Resampling.NEAREST)
        hr = self.transform(hr)
        lr = self.transform(lr)
        return {'hr' : hr, 'lr' : lr}