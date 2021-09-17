import cv2,sys,os,time,dlib
import numpy as np
import faceBlendCommon as fbc
from dataPath import DATA_PATH

FACE_DOWNSAMPLE_RATIO = 2
RESIZE_HEIGHT = 360

predictions2Label = {0:"No Glasses", 1:"With Glasses"}

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def prepareData(data):
  featureVectorLength = len(data[0])
  features = np.float32(data).reshape(-1,featureVectorLength)
  return features

def computeHOG(hog, data):
  hogData = []
  for image in data:
    hogFeatures = hog.compute(image)
    hogData.append(hogFeatures)

  return hogData


if __name__ == '__main__':

  # Load face detection and pose estimation models.
  modelPath = DATA_PATH + "models/shape_predictor_68_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(modelPath)

  # Initialize hog parameters
  winSize = (96,32)
  blockSize = (8,8)
  blockStride = (8,8)
  cellSize = (4,4)
  nbins = 9
  derivAperture = 0
  winSigma = 4.0
  histogramNormType = 1
  L2HysThreshold =  2.0000000000000001e-01
  gammaCorrection = 1
  nlevels = 64

  hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                          derivAperture,winSigma,histogramNormType,
                          L2HysThreshold,gammaCorrection,nlevels,1)

  # We will load the model again and test the model
  savedModel = cv2.ml.SVM_load("results/eyeGlassClassifierModel.yml")
  # Start webcam
  cap = cv2.VideoCapture(0)

  # Check if webcam opens
  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  while(1):
    try:
      t = time.time()
      # Read frame
      ret, frame = cap.read()
      height, width = frame.shape[:2]
      IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
      frame = cv2.resize(frame,None,
                         fx=1.0/IMAGE_RESIZE,
                         fy=1.0/IMAGE_RESIZE,
                         interpolation = cv2.INTER_LINEAR)

      landmarks = fbc.getLandmarks(detector, predictor, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), FACE_DOWNSAMPLE_RATIO)
      print("time for landmarks : {}".format(time.time() - t))
      #Get points from landmarks detector
      x1 = landmarks[0][0]
      x2 = landmarks[16][0]
      y1 = min(landmarks[24][1], landmarks[19][1])
      y2 = landmarks[29][1]

      cropped = frame[y1:y2,x1:x2,:]
      cropped = cv2.resize(cropped,(96, 32), interpolation = cv2.INTER_CUBIC)

      testHOG = computeHOG(hog, np.array([cropped]))
      testFeatures = prepareData(testHOG)
      predictions = svmPredict(savedModel, testFeatures)
      frameClone = np.copy(frame)
      #cv2.putText(frameClone, "Prediction = {}".format(predictions2Label[int(predictions[0])]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
      cv2.putText(frameClone, "Prediction = {}".format(predictions2Label[int(predictions[0])]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 4)
      
      print("Prediction = {}".format(predictions2Label[int(predictions[0])]))

      cv2.imshow("Original Frame", frameClone)
      cv2.imshow("Eye", cropped)
      if cv2.waitKey(1) & 0xFF == 27:
        break

      print("Total time : {}".format(time.time() - t))
    except Exception as e:
      frameClone = np.copy(frame)
      cv2.putText(frameClone, "Face Not detected properly", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
      cv2.imshow("Original Frame", frameClone)
#      cv2.imshow("Eye", cropped)
      if cv2.waitKey(1) & 0xFF == 27:
        break
      print(e)

  cv2.destroyAllWindows()
