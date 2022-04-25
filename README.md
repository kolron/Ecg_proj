# ECG4U
ECG4U is an image proccessing app designed to classify different types of arrhythmias from a 12-Lead ECG image.
The app recieves an image, classifies it to one of 9 types of arrhythmia and then explains the reasoning to said classification.

The app will provide a second opinion to doctors in real-time:
1. By classifiying, the app will help doctors in cases of uncertain diagnosis based on ECG and permit a faster route to an optimal treatment choice;
2. By explaining the reasoning, the will allow doctors to refine their skills in ECG reading and analysis, and imporving their long-term ability to treat patients.

Model was based on https://www.nature.com/articles/s41598-020-73060-w

## Usage
1. In order to simply classify an image - make sure you download [Classification_App](https://github.com/kolron/ECG4U/tree/main/Classification_App "Classification_App")and update the paths.
2. In order to re-train the network - please turn to [Training_And_Classification_App](https://github.com/kolron/ECG4U/tree/main/Training_And_Classification_App "Training_And_Classification_App"). Make sure you update the paths as well.

## Requirements
- Pytorch
- Numpy
- Pillow
- h5py

## Version 0.1
The app can only distinguish between normal sinus rhythm and atrial fibrillation (A-Fib).
Training is being still improved to make the accuracy even higher.

## Future implementations
1. Image perspective change imbedded in the learning process
2. Classification of other types of arrhythmias
3. Creating a UI
4. Explaining why the classification was made


## Bibliography
1. https://www.nature.com/articles/s41598-020-73060-w
2. https://github.com/Tereshchenkolab/paper-ecg
3. https://github.com/atabas/Heartbeat-Classification
4. https://static-content.springer.com/esm/art%3A10.1007%2Fs40846-021-00632-0/MediaObjects/40846_2021_632_MOESM1_ESM.pdf
5. http://2018.icbeb.org/Challenge.html
