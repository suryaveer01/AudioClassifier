import json
import torch
from torchvision import models, transforms
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

# Load the test image
img = Image(PilImage.open('cat.jpg').convert('RGB'))
# Load the class names
# with open('../data/images/imagenet_class_index.json', 'r') as read_file:
#     class_idx = json.load(read_file)
#     idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

idx2label = ['cat','dog','frog','horse','ship','truck']

# A ResNet Model
model = models.resnet34(pretrained=True)
# The preprocessing model
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

explainer = GradCAM(
    model=model,
    target_layer=model.layer4[-1],
    preprocess_function=preprocess
)
# Explain the top label
explanations = explainer.explain(img)
explanations.ipython_plot(index=0, class_names=idx2label)