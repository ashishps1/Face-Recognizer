#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()

mapping = {}

def get_images_and_labels(path):
    image_dirs = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    counter = 1

    for curr_dir in image_dirs:
        image_paths = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir)]

        for image_path in image_paths:
            print image_path
            # Read the image and convert to grayscale
            image_org = Image.open(image_path)
            image_pil = image_org.convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            image_show = np.array(image_org, 'uint8')
            # Get the label of the image

            nbr = counter
            mapping[nbr] = os.path.split(image_path)[1].split(".")[0]
            # Detect the face in the image
            faces = faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
                print "training image subject {}".format(nbr);
                cv2.imshow("Adding faces to traning set...", image_show[y: y + h, x: x + w])
                cv2.waitKey(50)
        counter += 1
    # return the images list and labels list
    return images, labels

# Path to train Yale Dataset
path_train = './faces/train'


# Path to test the Yale Dataset
path_test = './faces/test'

print " ***** TRAINING ***** \n\n"

images, labels = get_images_and_labels(path_train)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

'''
#testing in directories

print " ***** TESTING ***** \n\n"

image_dirs = [os.path.join(path_test, f) for f in os.listdir(path_test)]
counter = 1
for curr_dir in image_dirs:
    image_paths = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir)]
    for image_path in image_paths:
        predict_image_org = Image.open(image_path)
        predict_image_pil = predict_image_org.convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
        predict_image_show = np.array(predict_image_org, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            nbr_predicted = recognizer.predict(predict_image[y: y + h, x: x + w])
            nbr_actual = counter
            print "recognizing subject {}".format(nbr_actual)
            print nbr_actual,nbr_predicted
            if nbr_actual == nbr_predicted[0]:
                print "{} is Correctly Recognized ".format(nbr_actual)
            else:
                print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
            cv2.imshow("Recognizing Face", predict_image_show[y: y + h, x: x + w])
            cv2.waitKey(1000)
    counter += 1

'''

'''
#testing with an image containing multiple faces
path_test_img = './faces/testImages'
img_dir = [os.path.join(path_test_img, f) for f in os.listdir(path_test_img)]
i=0
while True:
    img = cv2.imread(img_dir[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(img_gray)

    for (x,y,w,h) in faces:
        nbr_predicted = recognizer.predict(img_gray[y: y + h, x: x + w])
        print "recognized subject is {}".format(mapping[nbr_predicted[0]])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,mapping[nbr_predicted[0]],(x,y+h+50), font, 0.5, (200,255,155), 2)
    cv2.imshow('img',img)
    k = cv2.waitKey(33)
    if k == ord('q'):
        break
    if k == 2555904:
        i+=1
    if k == 2424832:
        i-=1

    i=i%len(img_dir)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


#testing with a video containing moving faces
path_vid = './faces/testVideos/vid.mp4'
cap = cv2.VideoCapture(path_vid)

while(cap.isOpened()):
    ret, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(img_gray)
    count1=0
    count2=0
    for (x,y,w,h) in faces:
        count1 += 1
        nbr_predicted = recognizer.predict(img_gray[y: y + h, x: x + w])
        #print "recognized subject is {}".format(nbr_predicted[0])
        print "confidence is {}".format(nbr_predicted[1])
        confidence = int(nbr_predicted[1])
        if confidence<100:
            count2 += 1
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,mapping[nbr_predicted[0]],(x,y+h+50), font, 0.5, (200,255,155), 2)
    if count1!=0:
        print "Accuracy is {}".format((count2*1.0)/(count1))
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
