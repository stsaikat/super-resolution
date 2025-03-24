import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class RES(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3,3,5,padding=2,stride=1)
        # self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
    def forward(self, x):
        hx = x
        hx = self.conv(hx)
        # hx = self.bn(hx)
        hx = self.relu(hx)
        return hx
    
class SR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rebnconvlist1 = nn.ModuleList([RES() for _ in range(256)])
        self.relu1 = nn.ReLU()
        
    def upsample2x(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')
    
    def forward(self, x):
        hx = x
        hx = self.upsample2x(hx)
        return self.fix_upsample(hx)
    
    def fix_upsample(self, x):
        hx = x
        for rebnconv in self.rebnconvlist1:
            hx = checkpoint(rebnconv, hx)
            hx = torch.add(hx, x)
            hx = self.relu1(hx)
        
        hx = (hx - hx.min()) / (hx.max() - hx.min())
        return hx