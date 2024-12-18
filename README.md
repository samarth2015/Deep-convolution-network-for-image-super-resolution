# Deep-convolution-network-for-image-super-resolution
# SRCNN Implementation

## Dataset Preparation

For the training of the SRCNN model, the T91 image dataset was utilized. From each image, sub-images were extracted, processed, and used for training as described below:

### Steps to Generate Sub-images

1. **Convert to YCrCb format**:
   - Each image was converted to YCrCb format, and only the luminance channel (Y) was used.
   - This model can be extended to use the other channels as well.

2. **Extract sub-images**:
   - From the 91 images, sub-images of size 32x32 were extracted with a stride of 14, generating a total of 22,227 sub-images.

3. **Apply Gaussian blur**:
   - Gaussian blur was applied to each sub-image with a blur radius of 1.0.

4. **Downscale and upscale**:
   - Each blurred image was downscaled by a factor of 2 and subsequently upscaled by a factor of 2 using bicubic interpolation.

Below is the Python implementation for the preprocessing:

```python
import os
from PIL import Image, ImageFilter

# Directories
input_dir = 'archive/T91' # Replace with the path to T91 image directory
output_dir_lr = 'sub_images_lr' # Low-resolution output directory
output_dir_hr = 'sub_images_hr' # High-resolution output directory

# Parameters
sub_image_size = 32
stride = 14
scale = 2
blur_radius = 1.0

# Create output directories if not exist
os.makedirs(output_dir_lr, exist_ok=True)
os.makedirs(output_dir_hr, exist_ok=True)

# Generate sub-images
count = 0
for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)

    # Open image and convert to Y channel
    img = Image.open(filepath).convert('YCbCr').split()[0]
    h, w = img.size

    for i in range(0, h - sub_image_size + 1, stride):
        for j in range(0, w - sub_image_size + 1, stride):
            # Extract high-resolution sub-image
            hr_sub_image = img.crop((j, i, j + sub_image_size, i + sub_image_size))

            # Apply Gaussian blur
            hr_patch_image = hr_sub_image.filter(ImageFilter.GaussianBlur(blur_radius))

            # Downscale and upscale
            lr_patch_image = hr_patch_image.resize((sub_image_size // scale, sub_image_size // scale), Image.BICUBIC)
            lr_patch_image = lr_patch_image.resize((sub_image_size, sub_image_size), Image.BICUBIC)

            # Save sub-images
            lr_patch_image.save(os.path.join(output_dir_lr, f"{filename.split('.')[0]}_{i}_{j}.png"))
            hr_patch_image.save(os.path.join(output_dir_hr, f"{filename.split('.')[0]}_{i}_{j}.png"))

            count += 1

print(f"Total sub-images generated: {count}")
```

---

## Model Architecture

The SRCNN model comprises three convolutional layers:

1. **Patch Extraction and Representation (Layer 1):**
   - Input channel: 1 (using only the Y channel)
   - Kernel size: 9x9
   - Number of filters: 64
   - Activation: ReLU

2. **Non-linear Mapping (Layer 2):**
   - Input channel: 64
   - Kernel size: 1x1
   - Number of filters: 32
   - Activation: ReLU

3. **Reconstruction (Layer 3):**
   - Input channel: 32
   - Kernel size: 5x5
   - Number of filters: 1 (output channel)

### Mathematical Representation

- **Layer 1**:  
  `F1(Y) = ReLU(W1 * Y + B1)`  

  Where:  
  - `W1` and `B1` represent the filters and biases, respectively.  
  - Output: 64-dimensional feature map.

- **Layer 2**:  
  `F2(F1) = ReLU(W2 * F1 + B2)`  

  Where:  
  - `W2` and `B2` represent the filters and biases.  
  - Output: 32-dimensional feature map.

- **Layer 3**:  
  `F3(F2) = W3 * F2 + B3`  

  Where:  
  - `W3` and `B3` represent the filters and biases.  
  - Output: Super-resolved image.


### Model Implementation

```python
import torch
import torch.nn as nn
from torch.nn import init

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        # Patch extraction and representation (Layer 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Non-linear mapping (Layer 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Reconstruction (Layer 3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=0)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.conv1.weight, mean=0, std=0.001)
        init.normal_(self.conv2.weight, mean=0, std=0.001)
        init.normal_(self.conv3.weight, mean=0, std=0.001)
        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv2.bias, 0)
        init.constant_(self.conv3.bias, 0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
```

---

## Training the Model

During training, the Mean Squared Error (MSE) loss is calculated over the central `20X20` region of the output to avoid boundary artifacts. Below is the implementation:

```python
import torch.optim as optim

def train_model(model, dataloader, criterion, optimizer, num_epochs, save_interval=10, save_path="model_checkpoint.pth"):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Crop the central 20x20 region
            crop_size = 20
            _, _, h, w = outputs.size()
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            outputs_crop = outputs[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
            targets_crop = targets[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]

            # Compute loss
            loss = criterion(outputs_crop, targets_crop)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.10f}")

        # Save the model checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}.")

# Training configuration
criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 1e-5}
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```

## Results

### Training Metrics
- **Final MSE Loss:** 0.0000397035
- **Average PSNR on Training Set:** 52.32 dB

### Testing Metrics
The Set5 dataset was used for testing. The table below summarizes the performance for each image in the dataset.

| Image       | Scale | Sub-Images | Average PSNR (dB) |
|-------------|-------|------------|-------------------|
| Baby        | 2     | 1225       | 47.99503          |
| Bird        | 2     | 361        | 48.25294          |
| Butterfly   | 2     | 289        | 41.97383          |
| Head        | 2     | 324        | 48.39254          |
| Woman       | 2     | 345        | 57.00291          |

### Visualization of Filters
Filters from the first convolutional layer revealed meaningful patterns, such as edge detectors and Gaussian/Laplacian features.

![Filter visualizations](https://github.com/user-attachments/assets/fcd4a910-8432-4ce7-b9af-74f55c6e6ae1)


## Conclusion
This implementation of SRCNN demonstrates its capability to perform high-quality image super-resolution with promising results on benchmark datasets. The simplicity and effectiveness of the architecture make it a strong baseline for further research and enhancements.

