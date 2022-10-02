# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 02:21:09 2022

@author: General
"""
# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform data
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as nnf
import torchvision.models as models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device type is", device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

IMAGE_SIZE = 32
# First the image is resized then converted to a tensor

# transforms.Compose - composes multiple transforms together.
# transforms.Resieze - resize image to (x,y)
# transforms.ToTensor - convert PIL image or ndarray into Tensor.
# Tensors have a shape of (Col, Width, Height), value range of (0.0, 1.0).
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
							   transforms.ToTensor(),
							   transforms.Lambda(lambda x: x.to(device))])

# Label numbers for our images
is_chair_label = torch.tensor(0).to(device)
is_swivel_label = torch.tensor(1).to(device)
is_bed_label = torch.tensor(2).to(device)
is_sofa_label = torch.tensor(3).to(device)



def checkImage(Path, label):
	n_classes = 4
	model2 = models.resnet18(pretrained=True)
	model2.fc = nn.Linear(512, n_classes)
	model2.load_state_dict(torch.load( "model/trained_model.pt"))
	model2.to(device)
	model2.eval()

	test_img = Image.open(Path)
	test_img = composed(test_img)
	pred = model2(test_img.unsqueeze(0))
	prob = nnf.softmax(pred, dim=1)
	top_p, top_class = prob.topk(1, dim = 1)

	print(top_p.item())
	print(top_class)
	if top_class.item() == label:
		if top_p.item() >= 0.7:
			print("This is a valid label!")
			return(True)
		else:
			print("Not a good image")
			return(False)
	else:
		print("This is not a valid label!", f"Expected label is {top_class.item()}")
		return(False)