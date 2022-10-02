# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 21:04:34 2022

@author: General
"""
# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform data
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split as tts
import copy
import torch.nn.functional as nnf
import matplotlib.pylab as plt
import torchvision.models as models
# Python program to show time by perf_counter()
from time import perf_counter

"""
Set our device to use gpu if available.
Later called by <object>.to(device) to send a neural network or pytorch..
tensor to gpu memory and run any calculations on the gpu.
Comparing data from RAM and GPU memory will throw an exception.
"""
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




# Load our images from Data/stop and Data/not_stop
"""
We use os.listdir to extract filenames.
Images are not the same type so we use .convert('RGB') to keep images consistent
"""

# Label numbers for our images
is_chair_label = torch.tensor(0).to(device)
is_swivel_label = torch.tensor(1).to(device)
is_bed_label = torch.tensor(2).to(device)
is_sofa_label = torch.tensor(3).to(device)
#is_table_label = torch.tensor(4).to(device)

def load_images():
	# loop over the input images
	dataset = []

	# Check for images in images folder and label them 1
	for filename in os.listdir('col_chair'):
		image = Image.open(f'col_chair/{filename}')
		# Set hotdog to label 1 and not_hotdog to label 0
		image = composed(image)
		dataset.append([image, is_chair_label])

	# Check for images in bg folder and label them 0
	for filename in os.listdir('col_swivel'):
		image = Image.open(f'col_swivel/{filename}')
		# Set hotdog to label 1 and not_hotdog to label 0
		image = composed(image)
		dataset.append([image, is_swivel_label])

	# Check for images in bg folder and label them 0
	for filename in os.listdir('col_bed'):
		image = Image.open(f'col_bed/{filename}')
		# Set hotdog to label 1 and not_hotdog to label 0
		image = composed(image)
		dataset.append([image, is_bed_label])

	# Check for images in bg folder and label them 0
	for filename in os.listdir('col_sofa'):
		image = Image.open(f'col_sofa/{filename}')
		# Set hotdog to label 1 and not_hotdog to label 0
		image = composed(image)
		dataset.append([image, is_sofa_label])

# 	# Check for images in bg folder and label them 0
# 	for filename in os.listdir('col_table'):
# 		image = Image.open(f'col_table/{filename}')
# 		# Set hotdog to label 1 and not_hotdog to label 0
# 		image = composed(image)
# 		dataset.append([image, is_table_label])

	return dataset



# Plots our accuracy and loss for each iteration
def plot_stuff(COST,ACC):
	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.plot(COST, color = color)
	ax1.set_xlabel('Iteration', color = color)
	ax1.set_ylabel('total loss', color = color)
	ax1.tick_params(axis = 'y', color = color)

	ax2 = ax1.twinx()
	color = 'tab:blue'
	ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
	ax2.plot(ACC, color = color)
	ax2.tick_params(axis = 'y', color = color)
	fig.tight_layout()  # otherwise the right y-label is slightly clipped

	plt.show()



# Function that trains the model with data using defined hyperparameters
def train_model(scheduler,
				model,
				train_loader,
				validation_loader,
				criterion,
				optimizer,
				n_epochs,
				n_test,
				print_=True):
	loss_list = []
	accuracy_list = []
	correct = 0
	accuracy_best=0
	best_model_wts = copy.deepcopy(model.state_dict())


	#for epoch in tdqm(range(n_epochs)):
	for epoch in range(n_epochs):
		loss_sublist = []
		# Loop through the data in loader

		for x, y in train_loader:
			optimizer.zero_grad()
			# Use gpu for images
			x, y=x.to(device), y.to(device)
			# Set model in training mode
			model.train()
			# Calculate a prediction
			print(len(x))
			z = model(x)
			# Compare predication with actual value
			loss = criterion(z, y)
			loss_sublist.append(loss.data.item())
			# Propogates loss backwards through each layer
			loss.backward()
			# Use optimizer to change w and b values based on loss
			optimizer.step()

		print("epoch {} done".format(epoch))

		scheduler.step()
		loss_list.append(np.mean(loss_sublist))
		correct = 0

		# Check model prediction accuracy
		# Predicts an output and checks the most probable label vs actual label
		for x_test, y_test in validation_loader:
			x_test, y_test=x_test.to(device), y_test.to(device)
			model.eval()
			z = model(x_test)
			_, yhat = torch.max(z.data, 1)
			correct += (yhat == y_test).sum().item()
		accuracy = correct / n_test
		accuracy_list.append(accuracy)
		if accuracy>accuracy_best:
			accuracy_best=accuracy
			best_model_wts = copy.deepcopy(model.state_dict())


		if print_:
			print('learning rate',optimizer.param_groups[0]['lr'])
			print("The validaion  Cost for each epoch "+str(epoch + 1)+": "+str(np.mean(loss_sublist)))
			print("The validation accuracy for epoch "+str(epoch + 1)+": "+str(accuracy))
	model.load_state_dict(best_model_wts)
	return accuracy_list,loss_list, model



def main():

	# Use a pre-defined resnet18 model as our base, later training it to fit our..
	# data.
	"""
	Resnet18 is a model that uses residual learning to speed up training for
	models with many layers by passing the value of a node (residue) to a node 2
	layers down.
	https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#resnet
	"""
	model = models.resnet18(pretrained=True)


	# We will only train the last layer of the network, so we set the parameter..
	# requires_grad to False - the network is a fixed feature extractor.
	for param in model.parameters():
		param.requires_grad = False



	# Default for the last layer (model.fc) is:
	# (fc): Linear(in_features=512, out_features=1000, bias=True)
	# Since we only have 2 classes (stop or not_stop) we change this to:
	n_classes = 4
	model.fc = nn.Linear(512, n_classes)

	# Use gpu for model training and predictions
	model.to(device)

	# Get data and split it into test and train sets
	dataset = load_images()
	train_set, val_set = tts(dataset,train_size=0.9)
	# Save memory
	del dataset
	# Build our batches:
	batch_size=11
	train_loader = torch.utils.data.DataLoader(dataset=train_set,
											   batch_size=batch_size,shuffle=True)
	validation_loader= torch.utils.data.DataLoader(dataset=val_set ,
												   batch_size=1)


	# Define our hyperparams
	n_epochs=50
	lr=0.000001
	momentum=0.07
	# lr_scheduler: For each epoch change the learning rate between base_lr and..
	# max_lr. Allows to quickly train an accurate model without the need to..
	# manually change learning rate.
	lr_scheduler=True
	base_lr=0.001
	max_lr=0.01


	# Our optimizer and criterion
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	if lr_scheduler:
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
													  base_lr=base_lr,
													  max_lr=max_lr,
													  step_size_up=5,
													  mode="triangular2")



	# Train our model

	# Start the stopwatch / counter
	t1_start = perf_counter()

	# Train the model and retrieve loss and accuracy values
	accuracy_list,loss_list, model=train_model(scheduler,
											   model,
											   train_loader,
											   validation_loader,
											   criterion, optimizer,
											   n_epochs,
											   len(val_set))

	# Stop the stopwatch / counter
	t1_stop = perf_counter()
	elapsed_time = t1_stop - t1_start
	print("elapsed time", elapsed_time, "s")

	# Save the model to model.pt
	torch.save(model.state_dict(), 'trained_model.pt')

	# Plot loss and accuracy for each iteration
	plot_stuff(loss_list,accuracy_list)



if __name__ == "__main__":
	main()



