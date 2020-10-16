# Sign-Language-Detection
# About the Project
Sign Language Detector classifies the american sign language used by dumb and deaf so that they can easily communicate even with the people who don't know the language.It is an attempt to make the lives of differently abled easier.CNN is used to make classification.pyttx3 is used to convert text to voice so that visually impaired can also use it.
# Libraries used
This project makes the use of modules like:
Numpy
Tensorflow
OS
Keras
pyttsx3
SQLite
pickle
OpenCV 4.0

# Files Description
SET_HAND_HISTOGRAM.PY
This file has to be run in every new environment to set up the background settings for the model to work in. In this file, a screen pops up with green boxes, where you place your hand and press C (for capture), then the thresh screen comes up, where you can keep pressing C till you get a suitable threshold value. Once you find a suitable threshold value, you can save it by pressing S. You can use this file to create your own gestures or add more to our dataset that you can find here (*add link*). Our histogram values are saved in a file called ‘hist’ which is included in the repository.

CREATE_GESTURES.PY
This file was used to make a section of the dataset. Here when you run the file, they ask for the gesture ID (g_id) and the gesture label/text as the gesture name. If there is an already existing gesture (aka same g_id), the program asks if you want to rewrite it.
Once all the details are entered, the capture and thresh screens pop up, where you can hold and press S, and capture the gesture via the webcam. 1200 frames will be captured. We would suggest you to try slightly different angles with the same gestures for better training of the model. This program also calls rotate_gestures.py

ROTATE_GESTURES.PY
This file rotates and inverts the 1200 frames captured, to improve the quality of the dataset. This way, for every gesture, we have 2400 images in the end.

DISPLAY_GESTURES.PY
This file is used to display all the gestures in our dataset. It shows only one image for every gesture.

LOAD_IMAGES.PY
This file has to be used only when a new gesture is created. This program divides the entire image dataset into train_images, train_labels, test_images, test_labels, val_images and val_labels which are the training dataset, test dataset and validation dataset respectively along with their labels.

# Visuals
Gestures used:
![WhatsApp Image 2020-10-15 at 18 21 16](https://user-images.githubusercontent.com/72665043/96223519-b24ac180-0fab-11eb-903d-928ce4d4e56b.jpeg)

Loading of all the images:
![load images](https://user-images.githubusercontent.com/72665043/96223818-2ab18280-0fac-11eb-86ab-4ca197f99ea7.gif)



# Usage
This particular model can be used by differently abled to communicate with the ones who don't know sign language and vocalization of the sign language further enhace its usability by visually impared .
# Authors 
Prakriti Sharma  prakriti.s777@gmail.com
Vidushi Goyal    vgoyal_be19@thapar.edu
Harika Vattam   harikavattam@gmail.com
Snigdha         agarwalsnigdha418@gmail.com
