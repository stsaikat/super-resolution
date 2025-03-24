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
model.eval()

def inference():
    img = Image.open('small.jpg').convert('RGB')
    print('input size', img.size)
    img_tensor = to_tensor(img).to(device)
    # img_tensor = img_tensor[0,:,:]
    # img_tensor = img_tensor.unsqueeze(dim=0)
    img_tensor = img_tensor.unsqueeze(dim=0)
    with torch.no_grad():
        pred = model(img_tensor)

    print(pred.max(), pred.min())

    pred = pred.squeeze()
    pred = pred.detach().cpu()

    res = to_pil(pred).convert('RGB')
    res.save('out.jpg')
    print('output size', res.size)

    import cv2
    import numpy as np

    def calculate_psnr(image1_path, image2_path):
        # Read images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        # Ensure images have the same dimensions
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert images to float32 for precise calculations
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Compute Mean Squared Error (MSE)
        mse = np.mean((img1 - img2) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
        
        # Maximum pixel value (assuming 8-bit images)
        max_pixel = 255.0
        
        # Compute PSNR
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

    # Example usage:
    image1_path = "big.jpg"
    image2_path = "out.jpg"
    psnr_value = calculate_psnr(image1_path, image2_path)
    print(f"PSNR: {psnr_value} dB")
    

for _ in range(16):
    inference()