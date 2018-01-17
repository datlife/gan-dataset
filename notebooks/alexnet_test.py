# coding: utf-8
import json
import torch
import PIL
import numpy as np
import matplotlib.pyplot as plt
from alexnet import alexnet, preprocessor

use_gpu = torch.cuda.is_available()

# Define AlexNet Model and preprocessor
model  = alexnet(pretrained=True)
model.train(False) 
data_transforms = preprocessor() 

# Load ImageNet Labels Map
class_idx = json.load(open('./examples/imagenet_labels.json'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


raw    = PIL.Image.open('./examples/cute_cat.jpg').convert('RGB')
image  = data_transforms(raw)
inputs = torch.autograd.Variable(image, requires_grad=True).unsqueeze(0)

prediction = model(inputs)
prob, idx = torch.topk(prediction, 5)

# convert to numpy
prob = prob.data.tolist()[0]
idx = idx.data.tolist()[0]

softmax = np.exp(prob)/np.sum(np.exp(prob))
labels  = [idx2label[i] for i in idx]


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(raw, aspect="auto")
ax[1].barh(np.arange(5), softmax)

ax[0].axis('off')
ax[1].set_yticklabels(labels)
ax[1].invert_yaxis()

plt.show()

