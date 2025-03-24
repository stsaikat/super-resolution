import torch
from model import SR
import torchvision
from PIL import Image

to_pil = torchvision.transforms.ToPILImage()
to_tensor = torchvision.transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SR()
model.load_state_dict(torch.load('saved_models/model.pth'))
model = model.to(device)

img = Image.open('small.jpg').convert('RGB')
print('input size', img.size)
img_tensor = to_tensor(img).to(device)
img_tensor = img_tensor.unsqueeze(dim=0)

for _ in range(8):
    img_tensor = model.fix_upsample(img_tensor)
    
pred = model(img_tensor)

print(pred.max(), pred.min())

pred = pred.squeeze()
pred = pred.detach().cpu()

res = to_pil(pred).convert('RGB')
res.save('out.jpg')
print('output size', res.size)