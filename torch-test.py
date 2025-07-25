import urllib

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=2, out_channels=1, init_features=32, pretrained=False)

# Download an example image
# url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
#                  "TCGA_CS_4944.png")
filename_1 = "./C1-test.tif"
filename_2 = "./C2-test.tif"

# try:
#     urllib.URLopener().retrieve(url, filename)
# except:
#     urllib.request.urlretrieve(url, filename)

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

print(output[0])

# Display the thumbnail using Matplotlib
plt.imshow(output[0, 0])
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
