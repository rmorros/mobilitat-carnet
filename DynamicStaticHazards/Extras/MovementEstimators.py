from collections import defaultdict
from math import floor, sqrt, atan2, cos, sin

import cv2
import numpy as np


class LaneDetection:
    """
    Lane_Detection class basically detects the left and right line of an image
    """

    def __init__(self, resized_dims, cropped=True):
        self.cropped = cropped
        self.lines = None
        self.resized_dims = resized_dims
        self.roi = [(0, self.resized_dims[1]), (self.resized_dims[0] / 2, 0),
                    (self.resized_dims[0], self.resized_dims[1]), ]

    def region_of_interest(self, grey_scale_bool, img, vertices):
        """
        Masks a ROI in the image in order to reduce the area where detecting the lane
        """
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)

        # Create a match color with the same color channel counts.
        # If we are using grey_scale,just one color mask
        if grey_scale_bool:
            match_mask_color = 255

        else:
            # Retrieve the number of color channels of the image.
            channel_count = img.shape[2]
            match_mask_color = (255,) * channel_count

        # Fill inside the polygon
        cv2.fillPoly(mask, vertices, match_mask_color)

        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def makeLinePoints(self, y1, y2, line):
        """
        Convert a line represented in slope and intercept into pixel points
        :param y1: to compute the (x1, y1) point from line
        :param y2: to compute the (x1, y1) point from line
        :param line: slope and intercept that define the line
        :return:
        """
        if line is None:
            return None
        intercept, slope = line

        # cv2.line requires integers
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return (x1, y1), (x2, y2)

    def average_slope_intercept(self, lines):
        """
        Reduce all the lines detected into a combination of all of them, which are right and left lines.
        :param lines: bunch of lines detected from the Hough Filter
        :return: the left and right mostly accurate lane
        """
        avgLines = defaultdict(list)
        weights = defaultdict(list)
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # Ignore a vertical line
                slope = (y2 - y1) / float(x2 - x1)
                slope = floor(slope * 10) / 10

                # Discarting impossible slopes
                if slope == 0 or abs(slope) < 0.5 or abs(slope) > 2.5:
                    continue  # Avoid division by zero

                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                avgLines[slope].append((slope, intercept))
                weights[slope].append(length)

        keys = []
        for key in sorted(avgLines):
            keys.append(key)

        newAvgLines = defaultdict(list)
        newWeights = defaultdict(list)
        for i in range(1, len(keys)):
            if abs(keys[i] - keys[i - 1]) <= .1:
                slope = (keys[i] + keys[i - 1]) / 2.0
                for (s, intercept) in avgLines[keys[i]]:
                    newAvgLines[slope].append((s, intercept))
                for (s, intercept) in avgLines[keys[i - 1]]:
                    newAvgLines[slope].append((s, intercept))
                for (l) in weights[keys[i]]:
                    newWeights[slope].append((l))
                for (l) in weights[keys[i - 1]]:
                    newWeights[slope].append((l))
            else:
                if (i == 1):
                    slope = keys[i - 1]
                    for (s, intercept) in avgLines[keys[i - 1]]:
                        newAvgLines[slope].append((s, intercept))
                    for (l) in weights[keys[i - 1]]:
                        newWeights[slope].append((l))
                slope = keys[i]
                for (s, intercept) in avgLines[keys[i]]:
                    newAvgLines[slope].append((s, intercept))
                for (l) in weights[keys[i]]:
                    newWeights[slope].append((l))

        left = {}
        right = {}
        values_right = {}
        values_left = {}
        for key in newAvgLines:
            slope_mean = np.mean(list(map(lambda x: x[0], newAvgLines[key])))
            intercept_mean = np.mean(list(map(lambda x: x[1], newAvgLines[key])))
            if slope_mean > 0:
                left[slope_mean] = intercept_mean
                values_left[slope_mean] = np.size(newWeights[key])

            else:
                right[slope_mean] = intercept_mean
                values_right[slope_mean] = np.size(newWeights[key])

        if len(values_right) > 0:
            def_slope_right = np.dot(list(values_right.keys()), list(values_right.values())) / np.sum(
                list(values_right.values()))
            def_intercept_right = np.dot(list(right.values()), list(values_right.values())) / np.sum(
                list(values_right.values()))
        else:
            def_slope_right = None
            def_intercept_right = None

        if len(values_left) > 0:
            def_slope_left = np.dot(list(values_left.keys()), list(values_left.values())) / np.sum(
                list(values_left.values()))
            def_intercept_left = np.dot(list(left.values()), list(values_left.values())) / np.sum(
                list(values_left.values()))
        else:
            def_slope_left = None
            def_intercept_left = None

        return (def_intercept_left, def_slope_left), (def_intercept_right, def_slope_right)

    def laneLines(self, image, lines):
        """
        Get the points to draw the lines in the image
        :param image: image where we will draw the line
        :param lines: lines detected from the Hough Transformation
        :return: right and left lines defined by 2 points in determined Y coordinates
        """
        y1 = image.shape[0]  # Bottom of the image
        y2 = image.shape[0] * 0.5  # Slightly lower than the middle

        leftLane, rightLane = self.average_slope_intercept(lines)
        if leftLane[0] != None:
            leftLine = self.makeLinePoints(y1, y2, leftLane)
        else:
            leftLine = None

        if rightLane[0] != None:
            rightLine = self.makeLinePoints(y1, y2, rightLane)
        else:
            rightLine = None
        return leftLine, rightLine

    def drawLaneLines(self, image, lines, color=[0, 255, 0], thickness=20):
        """
        :param image: image where we draw the detected lines
        :param lines: 2 points left and right lines definition
        :param color: Color for the drawing --> Default: Blue
        :param thickness: Thick of the lines drawn --> Default: 20
        :return: the image where the lines are draw into as coordinate points
        """
        # Make a separate image to draw lines and combine with the orignal later
        lineImage = np.zeros_like(image)

        for line in lines:
            if line is not None:
                (x1, y1), (x2, y2) = line
                image = cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        # IF WANT THE LINES TO BE DRAWN SMOOTHLY
        # return lines, cv2.addWeighted(image, 1.0, lineImage, 0.95, 0.0)
        return lines, image

    def process_image(self, image):
        """
        Processes the frame passed as parameter to detect the lines left and right of the lane
        :param image: image where the lines are detected from
        :return: image with the detected lines drawn and the actual lines detected
        """

        self.frame = image
        original_img = np.copy(image)

        # Crop image with the region of interest
        if self.cropped:
            cropped_image = self.region_of_interest(False, original_img, np.array([self.roi], np.int32), )
            hsv = cv2.GaussianBlur(cropped_image, (5, 5), 0)
        else:
            hsv = cv2.GaussianBlur(original_img, (5, 5), 0)

        low_white = np.array([150, 150, 150])
        up_white = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, low_white, up_white)

        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40,
                                maxLineGap=25)
        if lines is None:
            return [], self.frame
        else:
            (left_lane, right_lane) = self.laneLines(self.frame, lines)
            lines_def = []
            if left_lane is not None:
                lines_def.append(left_lane)
            if right_lane is not None:
                lines_def.append(right_lane)
            return self.drawLaneLines(self.frame, lines_def)


class Distance:
    def __init__(self, dimensions, fps, width=2.3, lines=None, visualize=False):
        self.fps = fps
        self.width = width
        self.lines = lines
        self.warpRatio =1
        self.visualize = visualize
        self.resized_dims = dimensions
        self.objects = {}

    def getRealDistances(self, centroid):
        """
        Get the Distance in real world between the camera and the object
        :param box: coordinates for the box in the image
        :return: Distance X and Y between the object and the camera
        """
        # We need to obtain the centroid of the box of the object obtained to compute distance from it However we
        # think it is more interesting to compute the distance from the lower edge of the bounding box (as it is the
        # nearest to the ground)
        object_cx, object_cy = centroid
        distX, distY = self.projectiveModel((object_cx, object_cy))
        return (distX, distY)

    def defineLine(self, points):
        """
        Define a line by slope and intercept from two points
        :param points: 2 points of the line
        :return: slope and intercept for the line that cross both points
        """
        (x1, y1), (x2, y2) = points
        slope = floor((y2 - y1) / float(x2 - x1) * 10) / 10
        intercept = y1 - slope * x1
        line = (intercept, slope)
        return line

    def projectiveModel(self, objectCentroids):
        """
        Calculate Homogenous perspective to compute distances
        :param objectCentroids: centroid of the object bounding box coordinates
        :return: X and Y distance in meters of the object to the camera
        """
        point = self.calculateWrappingPoint(np.array(objectCentroids))
        centroid_camera = self.calculateWrappingPoint(np.array((self.resized_dims[0] / 2, self.resized_dims[1])))
        dist_pixelsX = abs(point[0] - centroid_camera[0])
        dist_pixelsY = abs(point[1] - centroid_camera[1])
        return self.warpRatio * dist_pixelsX, self.warpRatio * dist_pixelsY

    def getVanishingPoint(self, lines):
        """
        Computes the vanishing point for the lanes
        :param lines: left and right lines equations
        :return: point coordinates where left and right lines collide
        """
        (intercept_left, slope_left), (intercept_right, slope_right) = lines
        vanishingPointX = (intercept_left - intercept_right) / (slope_right - slope_left)
        vanishingPointY = intercept_left + slope_left * vanishingPointX
        vanishingPoint = (vanishingPointX, vanishingPointY)
        return vanishingPoint

    def birdEyeTransformation(self, image, option="Vanishing Point"):
        """
        Homogenous transformtion using Bird's eye view from OpenCv
        :param image: image where applying the transformation
        :param image_size: dimension of the image
        :param option: the method to implemente the Bird Eye transformation : by the already detected lines [default] or by using the vanishing point
        :return: It gives the Matrix of perspectives, the ratio and the image transformed
        """
        # Defining the inputs for the matrix
        (x1_l, y1_l), (x2_l, y2_l) = self.lines[0]
        (x1_r, y1_r), (x2_r, y2_r) = self.lines[1]
        IMAGE_W, IMAGE_H = self.resized_dims

        if option is None:
            # First approach: lane detection without vanishing point computation
            src = np.float32([[x1_l, y1_l], [x2_l, y2_l], [x1_r, y1_r], [x2_r, y2_r]])
            dst = np.float32([[0, IMAGE_H], [0, 0], [IMAGE_W, IMAGE_H], [IMAGE_W, 0]])

        elif option == "Vanishing Point":
            # Second approach: lane detection wit vanishing point computation
            left, right = self.defineLine(self.lines[0]), self.defineLine(self.lines[1])
            vanishingPoint = self.getVanishingPoint((left, right))
            warped_width = int(IMAGE_H * 0.15)
            top = vanishingPoint[1] + int(warped_width * 0.15)
            bottom = IMAGE_W + int(0.02 * IMAGE_W)

            def on_line(p1, p2, ycoord):
                return [p1[0] + (p2[0] - p1[0]) / float(p2[1] - p1[1]) * (ycoord - p1[1]), ycoord]

            p1 = [vanishingPoint[0] - warped_width / 2, top]
            p2 = [vanishingPoint[0] + warped_width / 2, top]
            p3 = on_line(p2, vanishingPoint, bottom)
            p4 = on_line(p1, vanishingPoint, bottom)
            src = np.float32([p1, p2, p3, p4])
            dst = np.float32([[0, 0], [IMAGE_W, 0], [IMAGE_W, IMAGE_H], [0, IMAGE_H]])

        self.M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
        self.Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

        # If we want to check accuracy of the method for bird eye transform point
        # self.checkAccuracy(src,dst)

        warped_img = cv2.warpPerspective(image, self.M, (IMAGE_W, IMAGE_H))
        if self.visualize == True:
            cv2.imshow("Bird Eye", warped_img)

        self.pointLeft = self.calculateWrappingPoint((x1_l, y1_l))
        self.pointRight = self.calculateWrappingPoint((x1_r, y1_r))
        pixel_distance = abs(self.pointLeft[0] - self.pointRight[0])
        self.warpRatio = self.width / pixel_distance

    def calculateWrappingPoint(self, point):
        """
        Calculate the coordinate in the bird's eye view of a point in the original image
        :param point: original image point
        :return: bird's eye image point
        """
        point = np.append(point, 1)
        a = self.M.dot(point)
        if a[2] == 0:
            dst = (0,0)
        else:
            dst = a / a[2]
        return np.array([dst[0], dst[1]])

    def calculateCameraPoint(self, point):
        """
        Calculate the coordinate in the camera's view of a bird's eye view point
        :param point: bird's eye image point
        :return: original image point
        """
        point = np.append(point, 1)
        a = self.Minv.dot(point)
        if a[2] == 0:
            dst = (0,0)
        else:
            dst = a / a[2]
        return np.array([dst[0], dst[1]])

    def checkAccuracy(self, src, dst):
        """
        Method to calculate the accuracy of the method to calculate the Transformation Matrix
        :param src: points to define the Matrix
        :param dst: points to output the Matrix
        :return: Accuracy of the perspective
        """
        if self.acc is None:
            self.acc = []
        i = 0
        diff = []
        for point in src:
            acc = np.linalg.norm(dst[i] - self.calculateWrappingPoint(point))
            diff.append(acc)
            i += 1
        self.acc.append(np.mean(diff))
        print("Accuracy is: " + str(np.mean(self.acc)))

    def makeLinePoints(self, y1, y2, line):
        """
        Convert a line represented in slope and intercept into pixel points
        :param y1: to compute the (x1, y1) point from line
        :param y2: to compute the (x1, y1) point from line
        :param line: slope and intercept that define the line
        :return:
        """
        if line is None:
            return None
        intercept, slope = line

        # cv2.line requires integers
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return (x1, y1), (x2, y2)

    def computeDistance(self, box):
        """
        Returns de real distance of the object to the camera
        :param box: Coordinates of an object --> Bounding box: ((x1, y1), (x2, y2))
        :return:  Real distance of the object to the camera
        """
        # We can just compute the distance if both lines are detected:
        if len(self.lines) == 2:
            (x1, y1, x2, y2) = box
            # We are actually searching for a point in the center X and bottom Y of the bounding box (as we measure just in the track lane)
            point = ((x1+x2)/2, max(y1, y2))
            real_obj_coordinates = self.getRealDistances(point)
            real_distance = np.linalg.norm(real_obj_coordinates)
            return real_distance, real_obj_coordinates
        print("Real distance of the object can no be find as lines have not been detected")

    def updateObject(self, id, boxes):
        """
        Updates the self.objects variable. This variable just contains the last 10 bounding boxes of an object to compute velocities and directions
        :param id: id of the object which box is going to be updated
        :param boxes: coordinates of the new bounding box detected for the object
        """
        # If object was never detected, initialize in dictionary of all objects detected
        past = self.objects.get(id)
        if past == None:
            self.objects[id] = []

        # In case the len of the object is 10, we need to update the oldest value with the ne one
        if len(self.objects[id]) == 10:
            del self.objects[id][0]

        # Save the box of the object detected for the last ten frames
        self.objects[id].append(boxes)

    def deleteObject(self, objectId):
        """
        Deletes an object inside the class when it disappears
        :param objectId: ID of the object that has disappeared
        """
        self.objects.pop(objectId, None)

    def computeAllDistances(self, image, lines, objects):
        """
        Main function for Distance class. Return all distances to all objects detected in an image
        :param image: image where the image have been detected to
        :param lines: left and right line detected. Necessary to compute ratios and scalate to real life measurements
        :param objects: all the centroid of the bounding boxes for all the objects detected with their ID's
        :return: the distances to all the centroid bounding boxes with its determined ID's
        """
        distances = {}
        coords = {}
        # Establish the new line
        if len(lines[0]) == 2:
            self.lines = lines

        if self.lines is None or len(self.lines) != 2:
            print("Lines not found, can not compute real distances.")
            for id, boxes in objects.items():
                # Update the object list array of boxes
                self.updateObject(id, boxes)
            return None, None
        # Establish the new homogeneous perspective matrix
        try:
            self.birdEyeTransformation(image)
        except Exception as ex:
            print("Lines not found, can not compute real distances.")
            for id, boxes in objects.items():
                # Update the object list array of boxes
                self.updateObject(id, boxes)
            return None, None

        for id, boxes in objects.items():
            # Update the object list array of boxes
            self.updateObject(id, boxes)
            # Compute the real distance for the specific object
            distances[id], coords[id] = self.computeDistance(boxes)

        return distances, coords

    def getCentroid(self, box):
        """
        Returns the Centroid coordinates of an object.
        :param box: coordinates of the bounding box of an object --> Box
        :return:
        """
        (x1, y1 , x2, y2) = box
        object_cx, object_cy = (x2 + x1) / 2, (y1 + y2) / 2
        return (object_cx, object_cy)


    def computeRelativeVelocity(self, objectId):
        """
        Compute the relative velocity of the object making some mathematical calculus on the last 10 bounding boxes of
        the object detected. The calculus are made in the bird eye's view.
        :param objectId: ID to identify the object in our intern self.objects dictionary
        :return: vector of the velocity, given by the module and the angle of the direction.
        """
        vectors = []
        boxes = self.objects[objectId]

        i=0
        if len(boxes) < 2:
            return None
        while i + 1 < len(boxes):
            # Transform all the the camera coordinates into the homogeneous perspective
            obj1 = self.calculateWrappingPoint(self.getCentroid(boxes[i]))
            obj2 = self.calculateWrappingPoint(self.getCentroid(boxes[i+1]))
            distance = [obj2[0] - obj1[0], obj2[1] - obj1[1]]
            # We normalize the vector
            dist_norm = sqrt(distance[0] ** 2 + distance[1] ** 2)
            vector_norm = dist_norm / self.fps
            angle = atan2(distance[1],distance[0])
            vectors.append((vector_norm, angle))
            i +=1
        # The output contains the module and angle for the Homogenous perspective
        return np.mean(np.array([norm[0] for norm in vectors])), np.mean(np.array([angle[1] for angle in vectors]))

    def getLast(self, objectId):
        """
        Used to obrain the last Box update of the Object Id
        :param objectId: the object we are referring to
        :return: the most updated bounding box for the object
        """
        list = self.objects[objectId]
        return list[len(list)-1]

    def computeVelocity(self, objectId, velocity_cam):
        """
        Compute the absolut velocity of the object taking into account the camera velocity
        :param objectId: ID to identify the object in our intern self.objects dictionary
        :param velocity_cam: velocity of the user
        :return: absolute velocity of the object
        """
        # Relative velocity that contains the module and angle for the Homogenous perspective
        orig_centroid = self.getCentroid(self.getLast(objectId))
        relative_object_velocity = self.computeRelativeVelocity(objectId)
        if relative_object_velocity is None:
            return None

        # Absolut velocity of the objects in the Homogenous perspective in vector coordinates
        absolute_object_velocity = self.final_point(velocity_cam, relative_object_velocity, factor=1/100)

        # Coordinates of the original centroid into the Bird's Eye View
        transformed_point =self.calculateWrappingPoint(orig_centroid)

        # Last point of the velocity array inside the Bird Eye's View
        final_point = self.final_point(transformed_point, absolute_object_velocity, factor = 1)

        #Last point of the velocity array inside the Camera's perspective
        cam_abs_velocity = self.calculateCameraPoint(final_point)

        # Computing the velocity array
        velocity = int(cam_abs_velocity[0] - orig_centroid[0]), int(cam_abs_velocity[1] - orig_centroid[1])

        # Computing the angle and the norm of the velocity
        angle = atan2(velocity[1], velocity[0])
        norm = sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        return (norm, angle)


    def computeAllVelocities(self, objectIds, cameraVelocity =25):
        """
         Return all velocities to all objects detected in an image
        :param objectIds:all the objects IDs
        :return: the velocity of all objects which their corresponding IDs
        """
        velocities = {}
        vector_camera = self.getCameraVelocity(cameraVelocity)
        for id in objectIds:
            # Compute the velocities for each object
            velocities[id] = self.computeVelocity(id, vector_camera)
        return velocities

    def getCameraVelocity(self, cameraVelocity):
        """
        Set the camera velocity vector. Take into account that it will always have the perpendicular direction to width of camera (0,1)*module
        :param cameraVelocity: modul of the velocity by default (if location used) or (norm, angle) of all mean static velocities obtained
        :return: vector of camera velocity
        """
        # If its only the module
        if type(cameraVelocity) == int:
            return (0, -1 * cameraVelocity)
        # If it contains module and norm, which means it is computed by static:
        else:
            return self.final_point((0,0),(-1* cameraVelocity[0], cameraVelocity[1]), factor=1/100)

    def drawCameraVector(self,camera_velocity_static):
        """
        Used to draw the velocity array of the camera
        :param camera_velocity_static: velocity array as coordinates in the homogenous view
        :return: initial point of the camera and final point. Used to draw the arrow
        """
        # We will also draw, in the camera view, the absolute velocity arrow for the camera.
        camera_abs_vel = self.getCameraVelocity(camera_velocity_static)
        initial_point_camera = (int((self.pointLeft[0] + self.pointRight[0]) / 2), int(self.resized_dims[1]))
        final_point_camera = (initial_point_camera[0] + camera_abs_vel[0], initial_point_camera[1] + camera_abs_vel[1])
        camera_array = self.calculateCameraPoint(final_point_camera)
        return initial_point_camera,(int(camera_array[0]), int(camera_array[1]))

    def final_point(self, point, velocity, factor=None):
        """
        From an initial point computes the velocity to the next point where should be placed if the velocity persits
        :param point: initial centroid of the object
        :param velocity: velocity of the object in this determined moment
        :return: output
        """
        if factor is None:
            factor = self.warpRatio
        norm, angle = velocity
        if factor == 0:
            return (point[0], point[1])
        x_mod = norm*cos(angle) / factor
        y_mod = norm*sin(angle) / factor
        return (point[0]+x_mod, point[1]+y_mod)
