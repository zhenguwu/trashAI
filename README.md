# trashAI

Project for DubHacks 2022 Environment Track

trashai.py runs a custom VGG16-based neural network to classify trash into 7 categories.
To use a predefined image rather than camera, run it with the image filename as an argument.
Images must be in testImages  

trashaiLite.py is the tflite-runtime version for raspberry pi

/models contains the trained models
/modelTraining contains the code and dataset used to create the CNN

Model is trained to 50 epochs and achieves 80% train acc and 70% val acc