import serial
import sys
import cv2
import time
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import pyrebase

interpreter = None
input_details = None
output_details = None
db = None

def setupFirebase():
    config = {
    "apiKey": "AIzaSyBP1SHQvDswK4r90_Q2k3pfRJjgj6SVMvI",
    "authDomain": "dubhacks2022-91ed5.firebaseapp.com",
    "databaseURL": "https://dubhacks2022-91ed5-default-rtdb.firebaseio.com",
    "projectId": "dubhacks2022-91ed5",
    "storageBucket": "dubhacks2022-91ed5.appspot.com",
    "messagingSenderId": "676659808821",
    "appId": "1:676659808821:web:a72d025e8cb75be3eabf7c",
    "measurementId": "G-SPEJ00K88K"
    }

    firebase = pyrebase.initialize_app(config)
    global db
    db = firebase.database()
    print("Firebase Successfully Intialized")

def setupTF():
    global interpreter
    interpreter = Interpreter(model_path="models/vgg16Model.tflite")
    interpreter.allocate_tensors()

    global input_details
    global output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def update_firebase_trash(value):
    childToUpdate = db.child("Items").child(value)
    oldValue = childToUpdate.get().val()
    db.child("Items").update({value: oldValue + 1})

def takePicture():
    camera = cv2.VideoCapture(0)
    time.sleep(0.5)
    return_value, image = camera.read()
    cv2.imwrite("capturedImages/capture.png", image)
    del(camera)

def identify_trash(imagePath):

    trashImg = cv2.imread(imagePath, -1)

    height, width, channels = trashImg.shape
    trashImg = trashImg[0:height, 0:height]
    trashImg = cv2.resize(trashImg,(224,224),interpolation = cv2.INTER_NEAREST)

    cv2.imshow('image',trashImg)
    cv2.waitKey(2000)

    trashImg = trashImg.astype(np.float32)
    trashImg /= 255.

    image_data = np.expand_dims(trashImg, 0)

    interpreter.set_tensor(input_details[0]['index'], image_data)


    time1 = time.time()
    interpreter.invoke()
    time2 = time.time()
    results = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classification_time = np.round(time2-time1, 3)
    print("Classificaiton Time =", classification_time, "seconds.")

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels("labels.txt")

    for i in top_k:
        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))


    return labels[top_k[0]]

def getSerial():
    ser = serial.Serial("COM1", 9600)
    while ser.in_waiting:
        print(ser.readline())
    return "b"
    

def update_motors(isTrash):
    ser = serial.Serial('/dev/ttyACM0', 9600)
    print(ser.name)
    if isTrash:
        ser.write(b'0')
    else:
        ser.write(b'1')

if __name__ == '__main__':
    setupTF()
    if len(sys.argv) > 1:
        path = "testImages/" + sys.argv[1]
        identify_trash(path)
    else:
        path = "capturedImages/capture.png"
        setupFirebase()
        while True:
            while True:
                serData = getSerial()
                if serData == "a":
                    takePicture()
                    result = identify_trash(path)
                    update_motors(result == "Trash")
                    update_motors(False)
                    try:
                        update_firebase_trash(result)
                    except:
                        print("Error connecting to Firebase")
