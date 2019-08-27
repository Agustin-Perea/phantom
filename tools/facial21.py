"""http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/"""
import cv2
import dlib
import glob
import numpy as np
import pickle

import math
import itertools
from sklearn.svm import SVC



emotions = ["anger", "disgust", "fear", "happiness", "neutral", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {}
tagged = []

#FLAG_TAG     = False
FLAG_SAVE    = False
FLAG_TRAIN   = False
FLAG_TEST    = True
PATH_TRAIN   = "."
#PATH_TAGFILE = "./tagged_faces.csv"
PATH_SVMFILE = "./model.pickle"
PATH_TEST    = "./test"  # point to a directory were you can easily check!
#CONST_FONT   = cv2.FONT_HERSHEY_SIMPLEX
PREDICTOR = pickle.load(open(PATH_SVMFILE,'rb'))


class TaggedFace:
    """
    Class to hold the results from the tagging.
    
    Formerly a named tuple, the __repr__ method was annoying on the console.

    :param tag: the emotion tag assigned
    :param path: path to the image file
    :param img: cv2/numpy array with the loaded image
    :param result: list of results of emotions in the image
    """
    def __init__(self, tag, result, path, img):
        self.tag = tag
        self.result = result
        self.path = path
        self.img = img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path},tag={self.tag})"


def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

def sortPred(path, pred):
    predictions = PREDICTOR.predict_proba(pred)
    resultsList = []

    for prediction in predictions:
        results = dict(zip(emotions, prediction))
        resultsList.append(results)

    Predinfo = dict(zip(path, resultsList))

    return Predinfo

def Tag(itm):
    image = cv2.imread(itm)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    get_landmarks(clahe_image)
    if data['landmarks_vectorised'] == "error":
        print("no face detected on this one")
    else:
        prediction = PREDICTOR.predict_proba([data['landmarks_vectorised']])
        results = dict(zip(emotions,prediction[0]))

        tag = max(results, key = lambda x: results.get(x) )

        tagged.append(TaggedFace(tag,results,itm,image))
    return None

def train():
    """
    Trains an SVM classifier based on the training data passed.
    """
    training_data = []
    training_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training = glob.glob(".\\%s\\*" %emotion)#get files from emotion directoy
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)#clahe_image
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
    
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)

    print("Training linear SVM...") #train SVM
    clf.fit(npar_train, training_labels)

    with open(PATH_SVMFILE, "wb") as filehandle:
        pickle.dump(clf, filehandle)
    return None

def test():
    prediction = glob.glob(f"{PATH_TEST}/*.jpg")
    for item in prediction:
            Tag(item)
  
    print(tagged)

    return None


def main():
   
    if FLAG_TRAIN:
        train()
    if FLAG_TEST:
        test()
        #if FLAG_SAVE:
        #    save_tagged(tagged, PATH_TAGFILE)


if __name__ == "__main__":
    main()
