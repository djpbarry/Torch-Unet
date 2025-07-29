import numpy as np
import tifffile  # Make sure to install this library using pip install tifffile
import torch
from PIL import Image
from torchvision import transforms

from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "Z:/working/barryd/hpc/python/Torch-Unet/crosstalk_detection_unet_2input_trained.pth"
model = UNet(in_channels=2, out_channels=1, init_features=32)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()
model.to(device)

filename_1 = "./C1-test-2.tif"
filename_2 = "./C2-test.tif"

input_image_1 = Image.open(filename_1)
input_image_2 = Image.open(filename_2)

m_1, s_1 = np.mean(input_image_1, axis=(0, 1)), np.std(input_image_1, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m_1, std=s_1),
])
input_tensor_1 = preprocess(input_image_1)

m_2, s_2 = np.mean(input_image_2, axis=(0, 1)), np.std(input_image_2, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m_2, std=s_2),
])
input_tensor_2 = preprocess(input_image_2)

input_batch = torch.cat([input_tensor_1, input_tensor_2], dim=0).unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Convert the output tensor to a numpy array
output_np = output.cpu().squeeze().numpy()

# Save the output as a TIF file
tifffile.imwrite('output_image-2.tif', output_np)

print("Output saved as output_image.tif")
