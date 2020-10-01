import time
from random import randint
import cv2
import json
import imutils
import numpy as np
from datetime import datetime
from ctypes import *
import matplotlib.pyplot as plt

import os
import sys

from Extras.MovementEstimators import LaneDetection
from Extras.MovementEstimators import Distance

from Tracker.CentroidTracker.centroidtracker2 import CentroidTracker
from Tracker.CentroidTracker.trackableobject2 import TrackableObject
from Tracker.motion_model import linear_motion
from Tracker.tracker import re3_tracker

import Detection.darknet as darknet


class FullSystem():
    # In this function we display all the functions that do not depend on the arguments
    def __init__(self):
        # Initialize variables
        self.trackableObjects = {}  # Dictionary to map each unique object ID to a TrackableObject
        self.detections = {}  # Dictionary to map each category for the app visualization
        self.colors = self.colors_array()
        self.total_colors = len(self.colors)
        self.dictionary = {}
        self.dictionary['Tracking'] = self.colors[(len(self.dictionary) % self.total_colors)]
        self.class_names = self.import_classnames()
        self.total_time_detection = 0  # to measure the total amount of time for detecting
        self.total_time_tracking = 0  # to measure the total amount of time for tracking
        self.total_time_centroid = 0  # to measure the total amount of time for centroid tracking
        self.total_time_distance_computing = 0  # to measure the total amount of time for distance computing
        self.total_time_lane_detection = 0  # to measure the total amount of time for the lane detection
        self.total_time_plotting = 0  # to measure the total amount of time for plotting video map distance

        self.static_objects = []
        self.static_names = ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.plotting = False

        # Parse arguments
        (self.input_path, self.output_path, self.resultsFile, self.visualize, self.maxDisappeared, self.maxDistance,
         self.skipFrames, self.scale, self.rotate, self.directory, self.distance, self.plotting) = self.argument_parsing()

        # There is only the possibility of plotting the distance when we have entered the possibility of using the distance
        if self.plotting == True and self.distance == False:
            print("We can not output the distances if 'Distance computation' is not enabled: set '--distance True'")
        self.plotting = self.plotting and self.distance

        # Get the video from the input fpath
        self.cap = cv2.VideoCapture(self.input_path)

        # Define models
        self.initializeYOLO()

        # VIDEO TRACKING
        # Initialize Re3 Tracker
        self.re3 = re3_tracker.Re3Tracker()

        # Initialize Centroid Tracker
        self.ct = CentroidTracker(maxDisappeared=self.maxDisappeared / self.scale,
                                  maxDistance=self.maxDistance / self.scale, visualize=self.visualize)

    def initializeYOLO(self):
        # global self.metaMain, self.netMain, self.altNames
        configPath = "./Detection/cfg/yolov4.cfg"
        weightPath = "./Detection/yolov4.weights"
        metaPath = "./Detection/cfg/coco.data"
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def colors_array(self):
        """
        Initialized an array of colors for the bounding boxes draw in the images
        :return: array of 20 random colors
        """
        colors = []
        for i in range(20):
            colors.append((randint(0, 256), randint(0, 256), randint(0, 256)))
        return colors

    def argument_parsing(self):
        """
        Parse all the arguments introduced by command line when executing the detection
        :return: arguments initialization
        """
        # HANDLING ARGUMENTS
        import argparse

        parser = argparse.ArgumentParser(description='Process a video.')
        parser.add_argument('-i', metavar='input', help='Path to source video', required=True)
        parser.add_argument('-o', '--output', type=str, default='output.mp4', help='Path to output saved video')
        parser.add_argument('--resultsFile', type=str, default='detection.txt',
                            help='File with all the information of the video processed and its detection')
        parser.add_argument('--visualize', type=bool, default=False,
                            help='Introduce True to display the video on real-time')
        parser.add_argument('--scale', type=int, default=3, help='Number to downsample the original video [default:1]')
        parser.add_argument('--maxDisappeared', type=int, default=20,
                            help='Assign maximum frames an object can not be detecting to stop its trakability')
        parser.add_argument('--maxDistance', type=int, default=200,
                            help='Assign maximum pixels distance from two centroids to be assign to different objects')
        parser.add_argument('--skipFrames', type=int, default=3,
                            help='Assign the skip frame for our algorithm (relation between detection and tracking interval)')
        parser.add_argument('--rotate', type=bool, default=False,
                            help='Introduce True if it is necessary to rotate the video')
        parser.add_argument('--directory', type=bool, default=False,
                            help='Introduce True if needs recursive detection. Remember to use input with directory')
        parser.add_argument('--distance', type=bool, default=False,
                            help='Introduce True if wants to display the distance of the objects to the camera')
        parser.add_argument('--plotting', type=bool, default=False,
                            help='Introduce True if wants to plot the distance in real time')

        args = parser.parse_args()
        return args.i, args.output, args.resultsFile, args.visualize, args.maxDisappeared, args.maxDistance, args.skipFrames, args.scale, args.rotate, args.directory, args.distance, args.plotting

    def write_video(self, video_in, output_path, scale):
        """
        Enables the script to start writing a video
        :param video_in: name of the video we are actually reading : used to get the proper dimensions
        :param output_path: name of the video to output
        :param scale: reduction of the dimensions: use to fasten the algorithm
        :return: the VideoWriter itself and some of its properties
        """
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.fps = int(video_in.get(cv2.CAP_PROP_FPS))
        size = (
        int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH) / scale), int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale))
        out = cv2.VideoWriter('out/' + str(output_path), fourcc, self.fps, size)
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        return (out, length, self.fps, size)

    def write_summary(self, date):
        """
        Write down the new detection of the video into the Summary file
        :param date: current time stamp for naming the directory
        :return: new name of the directory where video will be saved into
        """
        # open results file
        try:
            with open('out/Summary.txt', "r+") as fil:
                # first we read the file
                lines = fil.readlines()
                index = int(str(lines[-1]).split("\t")[0]) + 1
                print("Last index in summary file is " + str(index))
                towrite = "\n" + str(index) + "\t\t" + date + "\t\t" + self.input_path + "\t\t" + str(
                    self.skipFrames) + "\t\t\t\t" + str(self.maxDisappeared) + "\t\t\t\t" + str(self.maxDistance)
                fil.write(towrite)

        except IOError:
            print('File {} does not exist. We proceed on creating it'.format('out/Summary.txt'))
            f = open('out/Summary.txt', "x+")
            f.write("Index\tDate\t\t\t\tVideo input\t\t\t\t\tSkip Frames\t\tDisappeared\t\tMax Distance")
            index = 0
            towrite = "\n" + str(index) + "\t\t" + date + "\t\t" + self.input_path + "\t\t" + str(
                self.skipFrames) + "\t\t\t\t" + str(self.maxDisappeared) + "\t\t\t\t" + str(self.maxDistance)
            f.write(towrite)
        name = str(index) + "_" + date
        self.create_directory(name)
        time.sleep(1)
        return name

    def create_directory(self, name):
        """
        Create the directory where the output is going to be placed in
        :param name: name of the directory
        """
        try:
            os.makedirs("out/" + name)
            print("Directory " + name + " created successfully")
        except FileExistsError:
            print("Directory " + name + " already exists. We will introduce video in it")

    def write_detection(self, directory, configurations):
        """
        Enables writing down configurations and results of the detected video
        :param directory: name of the directory where document will be placed in
        :param configurations: text to write down
        :return:
        """
        try:
            f = open('out/' + directory + '/' + self.resultsFile, "x+")
            f.write(configurations)
        except:
            print("There was an error while creating file to write configurations of the detection")


    def import_classnames(self):
        """
        Import all the classes from the File classnames.txt
        :return: a variable with all the classes that the detector will actually detect
        """
        with open("Detection/data/classnames.txt") as f:
            lines = [line.rstrip() for line in f]
        return lines

    def count_objects(self):
        """
        Count the number of objects of each category detected in the video
        :return: dictionary with the amount ot objects of each category detected: {category: number}
        """
        count = {}
        for obj in self.trackableObjects.values():
            cat = obj.category.split(':')[0]
            if cat in count:
                count[cat] += 1
            else:
                count[cat] = 1
        return count

    def rotate_image(self, image, angle):
        """
        Rotate the image
        :param image: image to rotate
        :param angle: angle to rotate
        :return: resulting image
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.5)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def reformat_frame(self, frame, size):
        """
        Modify all parameters of the frame specified by arguments
        :param frame: image to be modified
        :param size: desired dimensions for the output image
        :return: resized and rotated image
        """
        # Re-scale the frame to the desired resolution
        resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        # Rotate image if necessary
        if self.rotate:
            image = self.rotate_image(resized_frame, 270)
            return image

        return resized_frame

    def write_initial_config(self, directory):
        """
        Initial configuration of the algorithm
        :param directory: where to be placed
        :return: text with the initial configurations
        """
        configurations = "DETECTION RESUME FOR VIDEO FILE " + self.input_path + "\n"
        configurations += "____________________________________________________________" + "\n"
        configurations += "For the detection in the video we are using the YOLOv4 Detector with yolov4.weigths in the COCO dataset." + "\n"
        configurations += "For the tracking in the video we are using the Re3 Tracker which will track objects detected from the Detector each 'skipFrames' parameter." + "\n"
        configurations += "For the correlative identification between Detector and Tracker we are using the Centroid Tracker from PyImageSearch with some modifications" + "\n"
        configurations += "____________________________________________________________" + "\n"
        configurations += "CONFIGURATIONS:" + "\n"
        configurations += "\t- Output video: " + directory + "/" + self.output_path + "\n"
        configurations += "\t- Output results file: " + directory + "/" + self.resultsFile + "\n"
        configurations += "\t- Visualitzation set as : " + str(self.visualize) + "\n"
        configurations += "\t- Skip Frames : " + str(self.skipFrames) + "\n"
        configurations += "\t- Max Disappeared : " + str(self.maxDisappeared) + "\n"
        configurations += "\t- Max Distance : " + str(self.maxDistance) + "\n"
        configurations += "\t- Scale : " + str(self.scale) + "\n"
        configurations += "\t- Distance computation : " + str(self.distance) + "\n"
        configurations += "\t- Plotting : " + str(self.plotting) + "\n"

        return configurations

    def write_final_config(self, configurations):
        """
        Final configuration of the algorithm
        :param configurations: text with the initial configurations
        :return: text with the total configurations
        """
        total_time =  self.total_time_plotting +  self.total_time_distance_computing + self.total_time_lane_detection  + self.total_time_tracking + self.total_time_detection + self.total_time_centroid
        # Preparing data for output results
        final_str = "Total amount of time for whole video detection " + "%0.4f" % self.total_time_detection + ' seconds: ' + "%0.4f" % (
                self.total_time_detection / total_time) + "\n"
        final_str += "Total amount of time for whole video tracking " + "%0.4f" % self.total_time_tracking + ' seconds: ' + "%0.4f" % (
                self.total_time_tracking / total_time) + "\n"
        final_str += "Total amount of time for whole video centroid tracking " + "%0.4f" % self.total_time_centroid + ' seconds: ' + "%0.4f" % (
                self.total_time_centroid / total_time) + "\n"
        final_str += "Total amount of time for whole video computing distances " + "%0.4f" % self.total_time_distance_computing + ' seconds: ' + "%0.4f" % (
                self.total_time_distance_computing / total_time) + "\n"
        final_str += "Total amount of time for whole video lane detection " + "%0.4f" % self.total_time_lane_detection + ' seconds: ' + "%0.4f" % (
                self.total_time_lane_detection / total_time) + "\n"
        final_str += "Total amount of time for whole video map distance plotting " + "%0.4f" % self.total_time_plotting + ' seconds: ' + "%0.4f" % (
                self.total_time_plotting / total_time) + "\n"
        final_str += "____________________________________________________________" + "\n"
        final_str += "Trackable objects: \n" + str(self.trackableObjects).replace(',', '\t').replace('{','\t').replace('}','')
        print(final_str, flush=True)
        configurations += "____________________________________________________________" + "\n"
        configurations += "MAP OF COLOURS: \n\t" + str(self.dictionary).replace('), ', ")\n\t").replace('{','').replace('}','\n')
        configurations += "____________________________________________________________" + "\n"
        configurations += "FINAL RESULTS:" + "\n"
        configurations += final_str
        configurations += "____________________________________________________________" + "\n"
        configurations += "COUNTING OBJECTS SEEN:" + "\n"
        configurations += str(self.count_objects()).replace(', ', "\n\t").replace('{', '\t').replace('}', '\n')
        return configurations

    def getRecursive(self):
        """
        If the argument parsing has detected the directory variable, it means that actually we are detecting all videos in a directory
        :return: Boolean that is True if detection is recursive, False if not
        """
        return self.directory

    def checkStatic(self, velocities):
        """
        Check all velocities and uses thresholding method to compute if object is static or not
        :param velocities: all velocities for the objects
        :return: set the boolean "static" option vairbles of the trackable objects to True or False depending on the Thresholding
        """
        threshold = 30
        for id, velocity in velocities.items():
            if velocity is not None:
                if velocity[0] > threshold:
                    self.trackableObjects[id].setOption("static", True)
                else:
                    self.trackableObjects[id].setOption("static", False)

    def setVelocity(self, categ):
        """
        This function is used to get the static mean velocities array of the static objects. Helped to compute camera's
        velocity as pixels in camera's perspective.
        :param categ: categories of the obtained centroid trackers
        :return:
        """
        static_velocity_modul = []
        static_velocity_angle = []

        for id, cat in categ.items():
            name = str(cat.split(':')[0])
            if name in self.static_names:
                velocity = self.distance_calculator.computeRelativeVelocity(id)
                if velocity is None:
                    return None
                static_velocity_modul.append(velocity[0])
                static_velocity_angle.append(velocity[1])

        if len(static_velocity_angle) > 0:
            return np.mean(static_velocity_modul), np.mean(static_velocity_angle)
        return None

    def plotMapDistance(self, directory, distances, frame_id, x_max=8, y_max=8):
        """
        We are creating a plot with 8 meters vertical and 8 meters horitzontal and display all the distances in it.
        :param distances: the distances for all the objects in this current frame
        :param frame_id: current number of the frame
        :return: plot
        """
        plt.figure(1)
        if frame_id != 0:
            plt.clf()
        plt.axis([-x_max / 2, x_max / 2, 0, y_max])
        plt.xlabel("X axis [meters]")
        plt.ylabel("Y axis [meters]")
        plt.title("Milliseconds of the video: " + str(self.fps * frame_id))
        plt.scatter(0, 0, c="b", marker='x')
        max_distance = np.linalg.norm((x_max / 2, y_max))

        for id, dist in distances.items():
            if (abs(dist[0]) < x_max / 2) and (abs(dist[1]) < y_max):
                category = str(self.trackableObjects[id].category)
                name = str(category.split(':')[0])
                if self.trackableObjects[id].centroids[-1][0] > self.size[0] / 2:
                    factor = 1
                else:
                    factor = -1
                abs_distance = np.linalg.norm(dist)
                plt.scatter(factor * dist[0], dist[1],
                            c=np.array([x / 256 for x in self.dictionary[name]]).reshape(1, -1), marker='o',
                            s=(1 / (abs_distance / max_distance)) ** 2)
                plt.annotate(name + " : " + str(id), (factor * dist[0], dist[1]), textcoords="offset points",
                             xytext=(0, 10),
                             ha='center')

        plt.tight_layout()
        plt.savefig("out/" + str(directory) + "/Plot/format_" + str(frame_id) + ".png")

    def createVideo(self, directory):
        image_folder = "out/" + str(directory) + "/Plot"
        video_name = "out/" + str(directory) + "/Map_Distance.mp4"

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        try:
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(video_name, fourcc, self.fps, (width, height))

            for image in images:
                video.write(cv2.imread(image_folder + "/" + str(image)))

            cv2.destroyAllWindows()
            video.release()
        except Exception as ex:
            print("No lines were detected so any distance could be computed")
        self.removeDirectory(image_folder)

    def removeDirectory(self, dir_path):
        try:
            files = [img for img in os.listdir(dir_path) if img.endswith(".png")]
            for f in files:
                try:
                    os.remove(dir_path + "/" + f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            os.rmdir(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

    def processVideo(self):
        """
        Main function of the Class: processes de Video, detects objects, classifies the objects and tracks them,
        computing all extras asked by arguments while executing
        """
        # Preparing output data results
        date = datetime.now().strftime("%Hh%Mm_%d%m%Y")
        directory = self.write_summary(date)
        configurations = self.write_initial_config(directory)

        # VIDEO PROCESSING
        # Exit if video not opened.
        if not self.cap.isOpened():
            print('Could not open {} video'.format(self.input_path))
            sys.exit()

        # handle argument -dont_show and save or not video
        print("Source Path:", self.input_path)
        if self.output_path:
            print("Video detected will not be displayed but saved in out/" + directory + "/" + self.output_path)
            (out, length, self.fps, self.size) = self.write_video(self.cap, directory + "/" + self.output_path,
                                                                  self.scale)
        else:
            print("Video detected displays in real-time")

        # Preparing output data results
        configurations += "\t- Frames per Second of input video : " + str(self.fps) + "\n"
        configurations += "\t- Resolution of input video : " + str(self.size) + "\n"
        configurations += "\t- Frame length of input video : " + str(length) + "\n"

        if self.distance == True:
            # Initialize Lane Detection Class and Distance Detection Class:
            self.lane_detector = LaneDetection(resized_dims=self.size)
            self.distance_calculator = Distance(dimensions=self.size, fps=self.fps)

        if self.plotting:
            self.create_directory(str(directory) + "/Plot")

        # VARIABLES OF USE
        id_number = []
        frame_id = 0

        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                           darknet.network_height(self.netMain), 3)
        factorX = self.size[0] / darknet.network_width(self.netMain)
        factorY = self.size[1] / darknet.network_height(self.netMain)
        # Loop over the video frames
        while (self.cap.isOpened()):
            status = "Waiting"
            r, frame = self.cap.read()

            if r:
                reformat_frame = self.reformat_frame(frame, self.size)
                rects = []
                categories = []
                static_frame_velocities = []
                # Only measure the time taken by YOLO and API Call overhead
                start_time = time.time()

                # DETECTION
                if frame_id % self.skipFrames == 0:
                    status = "Detecting"
                    id_number = []

                    frame_rgb = cv2.cvtColor(reformat_frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb,
                                               (darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain)),
                                               interpolation=cv2.INTER_LINEAR)

                    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

                    results = darknet.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.25)

                    self.total_time_detection = self.total_time_detection + time.time() - start_time

                    # Each time we made detection it is necessary to initialize new tracker:
                    start_time = time.time()
                    for cat, score, bounds in results:
                        name = str(cat.decode("utf-8"))
                        if name in self.class_names:
                            x, y, w, h = bounds
                            left = int(max(int(x - w / 2), 0) * factorX)
                            top = int(max(int(y + h / 2), 0) * factorY)
                            right = int(max(int(x + w / 2), 0) * factorX)
                            bottom = int(max(int(y - h / 2), 0) * factorY)

                            if name not in self.dictionary:
                                self.dictionary[name] = self.colors[(len(self.dictionary) % self.total_colors)]

                            # add new boxes detected for tracking
                            rects.append((left, bottom, right, top))
                            categories.append(name)

                            tracker = self.re3.track(name + ':' + str(len(id_number)), reformat_frame,
                                                     [left, bottom, right, top])
                            id_number.append(name + ':' + str(len(id_number)))
                    self.total_time_tracking = self.total_time_tracking + time.time() - start_time

                # TRACKING
                else:
                    status = "Tracking"

                    # grab the updated bounding box coordinates (if any) for each object that is being tracked
                    for a in id_number:
                        pos = self.re3.track(a, reformat_frame)

                        startX = int(pos[0])
                        startY = int(pos[1])
                        endX = int(pos[2])
                        endY = int(pos[3])
                        rects.append((startX, startY, endX, endY))
                        categories.append('Tracking')
                    self.total_time_tracking = self.total_time_tracking + time.time() - start_time

                # CENTROID TRACKING and EXTRAS parts
                start_time = time.time()

                # use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids
                objects, bboxes, disappeared, categ = self.ct.update(rects, id_number, status)
                self.total_time_centroid = self.total_time_centroid + time.time() - start_time

                if self.visualize:
                    print("Categories: " + str(categ))

                distances = None
                if self.distance:
                    # Lane detection
                    start_time = time.time()
                    none_original = np.copy(frame)
                    lines, drawn_lines_image = self.lane_detector.process_image(none_original)
                    self.total_time_lane_detection += time.time() - start_time
                    if lines is not None and len(lines) > 0:
                        # Distance and velocity computation
                        start_time = time.time()
                        if self.plotting:
                            distances, coord = self.distance_calculator.computeAllDistances(none_original, lines,
                                                                                            bboxes)
                        else:
                            distances, _ = self.distance_calculator.computeAllDistances(none_original, lines, bboxes)
                        if distances is not None:
                            current_velocity = self.setVelocity(categ)
                            if current_velocity is None:
                                current_velocity = 25
                            # Draw the arrow velocity for the camera
                            cam_pos, final_cam_pos = self.distance_calculator.drawCameraVector(current_velocity)
                            cv2.arrowedLine(reformat_frame, cam_pos, final_cam_pos, (255, 255, 255), thickness=2)
                            velocities = self.distance_calculator.computeAllVelocities(objects.keys(), current_velocity)
                            self.checkStatic(velocities)
                        self.total_time_distance_computing += time.time() - start_time

                # Centroid Tracker solution
                start_time = time.time()
                # loop over the tracked objects
                for (objectID, centroid) in objects.items():

                    # check to see if a traceable object exists for the current
                    # object ID
                    tobj = self.trackableObjects.get(objectID, None)

                    # if there is no existing traceable object, create one
                    if tobj is None:
                        tobj = TrackableObject(objectID, centroid, frame_id, str(categ[objectID]))

                    # otherwise, there is a traceable object so we can utilize it
                    # to determine direction
                    else:
                        if disappeared[objectID] <= 1:
                            tobj.centroids.append(centroid)
                            tobj.endFrame = frame_id

                        if disappeared[objectID] > 1:
                            try:
                                centroid = linear_motion(tobj.centroids[-2], tobj.centroids[-1])
                                tobj.centroids.append(centroid)
                            except IndexError:
                                pass

                    name = str(categ[objectID].split(':')[0])
                    cv2.rectangle(reformat_frame, (bboxes[objectID][0], bboxes[objectID][1]),
                                  (bboxes[objectID][2], bboxes[objectID][3]), self.dictionary[name], 2)
                    self.total_time_centroid = self.total_time_centroid + time.time() - start_time

                    start_time = time.time()
                    cv2.putText(reformat_frame, str(categ[objectID]), (bboxes[objectID][2], bboxes[objectID][3]),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, self.dictionary[name])

                    self.trackableObjects[objectID] = tobj

                    # Distance and velocity computation
                    if self.distance and distances is not None and velocities[objectID] is not None:
                        start_time = time.time()

                        # Drawing the velocity to each
                        final_centroid = self.distance_calculator.final_point(centroid, velocities[objectID], factor=1)
                        cv2.arrowedLine(reformat_frame, (int(centroid[0]), int(centroid[1])),
                                        (int(final_centroid[0]), int(final_centroid[1])), self.dictionary[name])
                        if self.visualize:
                            if objectID in distances.keys():
                                cv2.putText(reformat_frame, "Distance: " + str(distances[objectID]),
                                            (bboxes[objectID][2], bboxes[objectID][3] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                            0.5, self.dictionary[name])
                                if distances[objectID] < 2:
                                    print("Proximity alert: " + name + " " + str(objectID) + " is located at " + str(
                                        distances[objectID]) + " meters")
                            # Drawing if object is moving or not
                            task = "Static" if self.trackableObjects[objectID].getOption("static") else "Moving"
                            cv2.putText(reformat_frame, str(task), (bboxes[objectID][2], bboxes[objectID][3] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, self.dictionary[name])
                        self.total_time_distance_computing += time.time() - start_time

                # set_class_colors(colors, results, dictionary)
                if self.visualize:
                    print("STATUS: " + status + " -->" + str(self.trackableObjects), flush=True)

                if distances is not None:
                    if self.plotting:
                        start_time = time.time()
                        self.plotMapDistance(directory, coord, frame_id)
                        self.total_time_plotting += time.time() - start_time


                frame_id += 1
                if self.output_path:
                    out.write(reformat_frame)
                    percentage = float(frame_id / length) * 100
                    print("Percentage of frames detected: " + "%0.2f" % percentage + '%', flush=True)
                    if (percentage == 100):
                        configurations = self.write_final_config(configurations)
                        # Write data to output .txt files
                        self.write_detection(directory, configurations)

                        if self.plotting:
                            start_time = time.time()
                            self.createVideo(directory)
                            self.total_time_plotting += time.time() - start_time

                        self.cap.release()
                        out.release()
                else:
                    cv2.imshow("preview", reformat_frame)

            else:
                print("Finish detection")
                out.write(reformat_frame)
                configurations = self.write_final_config(configurations)
                # Write data to output .txt files
                self.write_detection(directory, configurations)
                if self.plotting:
                    start_time = time.time()
                    self.createVideo(directory)
                    self.total_time_plotting += time.time() - start_time
                self.cap.release()
                out.release()
                break
            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                break

        self.cap.release()
        out.release()

    def process_directory(self):
        """
        Main script if making recursive processing of videos for whole directory
        :return:
        """
        import glob
        path = self.input_path
        if str(self.input_path).endswith("/"):
            path = self.input_path
        else:
            path = self.input_path + "/"
        # List all .mp4 videos in the input directory path and its subdirectories
        files = [f for f in glob.glob(path + "**/*.mp4", recursive=True)]
        num_max = len(files)
        if num_max == 0:
            print("Error: directory introduced did not have any .mp4 files in it")
            sys.exit()
        else:
            print(
                "Files we will proceed to process:" + str(files).replace('[', '\n\t- ').replace(']', '\n').replace(',',
                                                                                                                   '\n\t- '))
            i = 0
            for f in files:
                print("We are currently processing video " + str(i) + " of " + str(num_max) + " : " + str(f))

                # Need to initialize the variables with the correct values again (avoid cache between videos)
                self.clear_cache(str(f))
                # Process the video
                self.processVideo()
                i += 1

    def clear_cache(self, file_name):
        """
        Function used to update the new initialization parameters when making recursive processive videos
        :param file_name: new video to process
        """
        self.input_path = file_name
        parts = file_name.split('/')
        self.output_path = parts[len(parts) - 1]
        # Initialize variables
        self.trackableObjects = {}  # Dictionary to map each unique object ID to a TrackableObject
        self.dictionary = {}
        self.dictionary['Tracking'] = self.colors[(len(self.dictionary) % self.total_colors)]

        self.total_time_detection = 0  # to measure the total amount of time for detecting
        self.total_time_tracking = 0  # to measure the total amount of time for tracking
        self.total_time_centroid = 0  # to measure the total amount of time for centroid tracking
        self.total_time_distance_computing = 0  # to measure the total amount of time for distance computing
        self.total_time_lane_detection = 0  # to measure the total amount of time for the lane detection
        self.total_time_plotting = 0  # to measure the total amount of time for plotting video map distance

        # Get the video from the input fpath
        self.cap = cv2.VideoCapture(self.input_path)

        '''
        # Define models
        # Load the Yolov4 Model: Model imports for configuration and weigths
        # get the video from the input fpath
        # VIDEO TRACKING
        # Initialize Re3 Tracker
        self.re3 = re3_tracker.Re3Tracker()
        '''
        # Initialize Centroid Tracker
        self.ct = CentroidTracker(maxDisappeared=self.maxDisappeared / self.scale,
                                  maxDistance=self.maxDistance / self.scale, visualize=self.visualize)


if __name__ == "__main__":
    fullsystem = FullSystem()
    # Check if we are actually asked for recursive Detection or just One vid detection
    if fullsystem.getRecursive():
        fullsystem.process_directory()
    else:
        fullsystem.processVideo()
