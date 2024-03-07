from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference
import torch
import math
import matplotlib.pyplot as plt

model = create_model("resnet34")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

plt.figure(figsize=(20,40))
plt.subplot(1,1,1)
image, mask = inference.inference(model, 'data/vinbigdata/images/train/0a1aef5326b7b24378c6692f7a454e52.png', 0.2)
plt.imshow(inference.img_with_masks( image, [mask[0], mask[1]], alpha = 0.1))
plt.show()