# AlexNet Implementation in PyTorch

![AlexNet ](/img/alexnet.png)

## Key characteristics of this AlexNet implementation:

### 1. **Architecture**:
- 5 convolutional layers
- 3 max pooling layers
- 3 fully connected layers
- Input size: 224×224×3 (RGB images)
- Output: 1000 classes (ImageNet classes)

### 2. **Innovative Features from 2012**:
- ReLU activation instead of tanh/sigmoid
- Local Response Normalization (LRN)
- Dropout (0.5) in fully connected layers
- Large convolutional kernels (11×11) in the first layer
- Overlapping pooling

### 3. **Default Parameters**:
- stride=4 in the first conv layer
- 96 filters in the first layer (originally split across 2 GPUs)
- Final output of 1000 classes for ImageNet

 https://colab.research.google.com/drive/1v-DRh_MX2xHb4ePzciJbbZ0eVbNW2iQJ?usp=sharing

