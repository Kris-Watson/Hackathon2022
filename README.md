# Hack'O Mania 2022

With the advance of Machine Learning software and leaps in computing hardware, data analysis has never been easier. However, obtaining specific or new clean data to train models, is a manual and time consuming process.

We have designed a mobile app mockup that allows buisnesses to request a particular dataset from users. Here is how it works:
1) Users to take a picture of the object. The mobile app will ensure that image data is relatively consistent, perfectly cropped and not a duplicate across various users.
2) The user chooses a label that best fits their picture (eg: blue or green or yellow car).
3) We use a model which has been trained with a small but highly varied dataset to make sure the image is not wildly off what we expect. Reject images which are not good.
4) Send the data to a database and reward the user with whatever payment method is viable.
5) In the case of the user's image being incorrectly rejected, user may appeal - allowing a person to check the validity of the image.



# Code
Python Model that checks user input data and checks if it meets requirements (clean image, accurately labelled and unique).

Currently using a preset dataset containing images of chairs, beds, sofas and sviewel chairs.

1) Train the model using our dataset by running train_model.py. Model is stored in "trained_model.pt"
2) Run main.py to check the validity of the image in "myPath"
3) In deployment, data would then be sent to a database for client use.


To use a different dataset, change the "load_dataset" class (in train.py) and labels (in train.py and main.py) and run the test again. 

Dummy dataset (prev_images) present to check unique-ness of data.
Use test.py to run all checks on user data.
