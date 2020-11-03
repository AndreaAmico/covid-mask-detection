# Face mask detection

This repository includes two different machine learning-based approaches to detect from a video stream the presence of people wearing and not wearing the face mask to fight the spread of the covid-19 pandemic. The first approach is lightweight, mostly suitable for edge computing applications. The latter is instead computationally intensive and requires dedicated hardware such as a local or cloud GPU. Here we choose a cloud approach employing AWS Sagemaker.

![Live demo](https://media3.giphy.com/media/y0gpT045GYWWbU9fhl/giphy.gif)

### edge-coral
Edge computing mask detection approach. The script has been tested on a Raspberry 3b using the Coral USB accelerator. 

Requirements:
```
pip install tensorflow
pip install opencv-contrib-python
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl 
```

Coral dependencies: 
```
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get install libedgetpu1-std
```

files:
- coral_test.py: mask detection script for raspberry pi
- pc-test: computer version of the mask detection model (not raspberry)
- train-model: mask lightweight classifier definition and training.

### AWS
files:
- train.py: define and train the fast R-CNN model
- generate.py: create an AWS endpoint to deploy the model
- train_and_deploy.py: jupyter notebook to invoke train.py and generate.py with AWS authentication.

- **TODO** aws-kinesis ->  Employ OpenCV to send generate a video stream for AWS.

