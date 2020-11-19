from tkinter import *
from tkinter import filedialog

import social_distancing_config as config
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from utils import backbone
import object_counting_api

a = Tk()

a.geometry("500x300")
a.config(bg='#A0A2A3')
a.title('Crowd Check')

text1 = ""


def fopen():
    file1 = filedialog.askopenfile()
    entryText.set(file1.name)

    print(file1.name)
    global text1
    text1 = file1.name


def detector():
    # construct the argument parse and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
    #ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
    #ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
    #args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(text1)
    #writer = None
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_movie = cv2.VideoWriter('output_sdd.avi', fourcc, fps, (width,height))

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        # check to see if the output frame should be displayed to our
        # screen
        #if args["display"] > 0:
            # show the output frame
        output_movie.write(frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    vs.release()
    output_movie.release()
    cv2.destroyAllWindows()

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        #if writer is None:
            # initialize our video writer
            #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                     #(frame.shape[1], frame.shape[0]), True)


        # if the video writer is not None, write the frame to the output
        # video file
        #if writer is not None:
        #output_movie.write(frame)

# people counting on video input
def counter():

    input_video = text1

    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

    targeted_objects = "person"  # (for counting targeted objects)
    is_color_recognition_enabled = 0

    object_counting_api.targeted_object_counting(input_video, detection_graph, category_index,
                                                 is_color_recognition_enabled,
                                                 targeted_objects)  # targeted objects counting

# people counting in real-time
def real_time():
    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')
    targeted_objects = "person"
    is_color_recognition_enabled = 0

    object_counting_api.object_counting_webcam(detection_graph, category_index, is_color_recognition_enabled,
                                               targeted_objects)


Label(text="CROWD CHECK", font='Helvetica 20 bold', bg='#A0A2A3').place(x=140, y=20)
button1 = Button(text="Select Video", padx=50, command=fopen).place(x=30, y=80)
Label(text="Video Path", bg='#A0A2A3').place(x=30, y=115)
entryText = StringVar()
Entry(textvariable=entryText, width=30).place(x=110, y=115)
button2 = Button(text="Social Distancing Detector", padx=50, command=detector).place(x=30, y=160)
button3 = Button(text='People Counter', padx=50, command=counter).place(x=30, y=205)
button4 = Button(text="Real-time Counter", padx=50, command=real_time).place(x=30, y=250)

a.mainloop()
