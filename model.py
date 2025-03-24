import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONVADDRE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_s1 = nn.Conv2d(3,3,5,padding=2,stride=1)
        self.bn_s1 = nn.BatchNorm2d(3)
        self.relu_s1 = nn.ReLU()
    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout
    
class SR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rebnconvlist1 = nn.ModuleList([REBNCONVADDRE() for _ in range(16)])
        self.relu1 = nn.ReLU()
        self.rebnconvlist2 = nn.ModuleList([REBNCONVADDRE() for _ in range(16)])
        self.relu2 = nn.ReLU()
        
    def upsample2x(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')
    
    def forward(self, x):
        hx = x
        hx = self.upsample2x(hx)
        return self.fix_upsample(hx)
    
    def fix_upsample(self, x):
        hx = x
        for rebnconv in self.rebnconvlist1:
            hx = rebnconv(hx)
        hx = torch.add(hx, x)
        hx = self.relu1(hx)
        
        for rebnconv in self.rebnconvlist2:
            hx = rebnconv(hx)
        hx = torch.add(hx, x)
        hx = self.relu2(hx)
        
        hx = (hx - hx.min()) / (hx.max() - hx.min())
        return hx