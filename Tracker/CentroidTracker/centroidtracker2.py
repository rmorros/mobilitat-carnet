# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import sys

class CentroidTracker:
        def __init__(self, maxDisappeared=50, maxDistance=100, visualize = False):
                # initialize the next unique object ID along with two ordered
                # dictionaries used to keep track of mapping a given object
                # ID to its centroid and number of consecutive frames it has
                # been marked as "disappeared", respectively
                self.nextObjectID = 0
                self.objects = OrderedDict()
                self.disappeared = OrderedDict()

                # JRMR
                self.rectangles = OrderedDict()
                self.visualize  = visualize

                # EVM
                self.categories = OrderedDict()
                
                # store the number of maximum consecutive frames a given
                # object is allowed to be marked as "disappeared" until we
                # need to deregister the object from tracking
                self.maxDisappeared = maxDisappeared

                # store the maximum distance between centroids to associate
                # an object -- if the distance is larger than this maximum
                # distance we'll start to mark the object as "disappeared"
                self.maxDistance = maxDistance

        def register(self, centroid, category, rect = None):  # JRMR: added 'rect' parameter      #EVM: added 'category' parameter

                # when registering an object we use the next available object
                # ID to store the centroid
                self.objects[self.nextObjectID] = centroid

                if self.visualize == True:    # JRMR
                        print ('CentroidTracker: registering ID {}'.format(self.nextObjectID))

                # JRMR & EVM
                if rect != None:
                        self.rectangles[self.nextObjectID] = rect
                        self.categories[self.nextObjectID] = category

                self.disappeared[self.nextObjectID] = 0
                self.nextObjectID += 1

        def deregister(self, objectID):
                # to deregister an object ID we delete the object ID from
                # both of our respective dictionaries
                del self.objects[objectID]
                del self.disappeared[objectID]
                
                # JRMR & EVM
                if objectID in self.rectangles:
                        del self.rectangles[objectID]
                        del self.categories[objectID]

        # EVM added 'categories' and 'status' parameters
            # categories cains a list of the current categories detected for each rects
            # status contains the method of prediction of these rects: 'detection' or 'tracking'
        def update(self, rects, new_categories, status):
            # check to see if the list of input bounding box rectangles is empty
            if len(rects) == 0:
            
                    if self.visualize == True:
                            print ('CentroidTracker: 0 rects')
                            
                    # loop over any existing tracked objects and mark them as disappeared
                    for objectID in list(self.disappeared.keys()):
                            self.disappeared[objectID] += 1

                            # if we have reached a maximum number of consecutive frames where a given
                            # object has been marked as missing, deregister it
                            if self.disappeared[objectID] > self.maxDisappeared:
                                    if self.visualize == True:
                                            print ('Deregistering object {} after {} frames'.format(objectID, self.maxDisappeared))
                                            sys.stdout.flush()
                                    self.deregister(objectID)

                    # return early as there are no centroids or tracking info to update
                    new_names={}
                    for id, cat in self.categories.items():
                        new_names[id]=(cat.split(':')[0]+":" +str(id))
                    return (self.objects, self.rectangles, self.disappeared, new_names)   #EVM added 'categories'


            # initialize an array of input centroids for the current frame
            inputCentroids  = np.zeros((len(rects), 2), dtype="int")
            inputRectangles = np.empty((len(rects)), dtype=object)   # JRMR
            objectIDs = list(self.objects.keys())

            
            # loop over the bounding box rectangles
            for (ii, (startX, startY, endX, endY)) in enumerate(rects):
                    # use the bounding box coordinates to derive the centroid
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    # cY = endY
                    inputCentroids[ii]  = (cX, cY)
                    inputRectangles[ii] = (startX, startY, endX, endY)  # JRMR

                    
            # if we are currently not tracking any objects take the input centroids and register each of them
            if len(self.objects) == 0:
                    
                    for jj in range(0, len(inputCentroids)):
                            self.register(inputCentroids[jj], new_categories[jj],  inputRectangles[jj]) # JRMR: added inputRectangles argument
                                                                                                    # EVM : added categories argument
                    
            # otherwise, are are currently tracking objects so we need to try to match the input centroids to existing object centroids
            else:
                # EVM: if we are in tracking status, we give priority to centroids with same Category and ID (tehrerfore, from the same object)
                if status == "Tracking":
                    usedIndexes = set()
                    deregister = []
                    # Loop all previous categories stored in the CT
                    if self.visualize == True:
                        print("TRACKING:", flush=True)
                        print("SELF.CATEGORIES:"+ str(self.categories), flush=True)
                        print("NEW CATEGORIES:"+ str(new_categories), flush=True)

                    for id, cat in self.categories.items():
                        # If both new and previous categories+ID match, then we check if it does not overpass max_distance. Then, we coordinate centroids.
                        if cat in new_categories:
                            index = new_categories.index(cat)
                            ''' WE DO NOT TAKE INTO ACCOUNT SURPASS OF MAX_DISTANCE : if names match means its the same
                            #If it overpasses max_distance, object not added
                            if dist.cdist(np.array(self.objects[id]), inputCentroids[index]) > maxDistance
                                if self.visualize == True:
                                    print ('CentroidTracker: ID {} not matched to rectangle {} because D={} > maxDistance = {}'.format(objectIDs[row], inputRectangles[col], D[row,col], self.maxDistance))
                                    unusedIds.add(id)
                                continue
                            '''
                            self.objects[id]     = inputCentroids[index]
                            self.rectangles[id]  = inputRectangles[index] # JRMR
                            self.disappeared[id] = 0
                            usedIndexes.add(index)
                        # Else means that the category was predicted before but not now, which means that we mark it as disappeared.
                        else:
                            if self.visualize == True:
                                print ('CentroidTracker: ID {} marked as disappeared'.format(id))

                            # grab the object ID for the corresponding row
                            # index and increment the disappeared counter
                            self.disappeared[id] += 1

                            # check to see if the number of consecutive
                            # frames the object has been marked "disappeared"
                            # for warrants deregistering the object
                            if self.disappeared[id] > self.maxDisappeared:
                                if self.visualize == True:
                                    print ('Deregistering ID {} after {} frames'.format(id, self.maxDisappeared))
                                    sys.stdout.flush()
                                deregister.append(id)
                    
                    for id in deregister:
                        self.deregister(id)

                    '''
                    # Unused Indexes are all indexes from new catgeories that have not been match, so need to register:
                    unusedIndexes = set(range(0, len(new_categories))).difference(usedIndexes)
                    for ind in unusedIndexes:
                        self.register(inputCentroids[ind], new_categories[ind], inputRectangles[ind])
                    '''
                # DETECTION STATUTS
                else:
                    objectCentroids = list(self.objects.values())
                    used_indexes = set()
                    #We split the category format "categorynames:id" into a list of just categorynames
                    new_names = list(map(lambda x : str(x).split(':')[0], new_categories))
                    old_names = list(map(lambda x : str(x).split(':')[0], self.categories.values()))
                    if self.visualize == True:
                        print("OLD NAMES: "+ str(old_names), flush=True)
                        print("NEW NAMES: "+ str(new_names), flush=True)

                    # Check if there are new classes nerver detected before. If so, register them.
                    different = list(filter(lambda x: x not in old_names, new_names))
                    if len(different) > 0:
                        index_diff = list(map(lambda x : new_names.index(x), different))
                        if self.visualize == True:
                            print("DIFFERENT: "+ str(different), flush=True)
                        for i in index_diff:
                            # We register this new Centroids to the CT
                            self.register(inputCentroids[i], new_categories[i], inputRectangles[i])
                            used_indexes.add(i)

                    # We have to redefine lists of ObjectIDs, ObjectCentroid, InputCentroids and InputRectangles
                    # that have the same COCO class name

                    # AS we have already computed the different classes from new_categories, now we will be looping the old_names
                    # Take into account that Different list has items from new_names that are not in old_names.
                    accumulatedDisappeareance = 0
                    for category in set(old_names):
                        # Categories from Old_names which are not in new_names --> mark as dissappeared
                        if category not in new_names: 
                            indices =  [i for i, x in enumerate(old_names) if x==category]
                            if self.visualize == True:
                                print("Category " +str(category)+" in positions "+ str(indices)+ ", which are not in new_names --> mark as dissappeared ")
                            for id in indices:  
                                objectID = list(self.categories.keys())[id-accumulatedDisappeareance]

                                if self.visualize == True:
                                    print ('\tCentroidTracker: ID {} marked as disappeared'.format(objectID))
                                

                                # grab the object ID for the corresponding row
                                # index and increment the disappeared counter
                                self.disappeared[objectID] += 1

                                # check to see if the number of consecutive
                                # frames the object has been marked "disappeared"
                                # for warrants deregistering the object
                                if self.disappeared[objectID] > self.maxDisappeared:
                                    if self.visualize == True:
                                        print ('\t\tDeregistering ID {} after {} frames'.format(objectID, self.maxDisappeared))
                                        sys.stdout.flush()
                                    self.deregister(objectID)
                                    accumulatedDisappeareance +=1
                        
                        # Categories find on both new and old
                        else:
                            # Obtain a list of indexes with the current position of each element of this category
                            indices_old =  [i for i, x in enumerate(old_names) if x==category]   
                            indices_new =  [i for i, x in enumerate(new_names) if x==category]  
                            objCent = list(map(lambda x: objectCentroids[x], indices_old)) 
                            inCent = list(map(lambda x: inputCentroids[x], indices_new))
                            inRect = list(map(lambda x: inputRectangles[x], indices_new))
                            if self.visualize == True:
                                print("\tCATEGORY: "+ str(category), flush=True)
                                print("\t\tINDICES OLD: "+ str(indices_old), flush=True)
                                print("\t\tINDICES NEW: "+ str(indices_new), flush=True)
                                print("\t\tOBJECT CENTROIDS: "+ str(objCent), flush=True)
                                print("\t\tINPUT CENTROIDS: "+ str(inCent), flush=True)
                            acc=self.update_distance(indices_old, objCent,inCent, inRect,category, accumulatedDisappeareance)
                            accumulatedDisappeareance += acc

            new_names={}
            for id, cat in self.categories.items():
                new_names[id]=(cat.split(':')[0]+":" +str(id))
            # return the set of trackable objects
            return (self.objects, self.rectangles, self.disappeared, new_names)  # EVM: added output "new_names"

        def update_distance(self, objectIDs, objectCentroids, inputCentroids, inputRectangles, category, accumulatedDisappeareance):

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                    
                    # if we have already examined either the row or
                    # column value before, ignore it
                    if row in usedRows or col in usedCols:
                        continue

                    # if the distance between centroids is greater than
                    # the maximum distance, do not associate the two
                    # centroids to the same object
                    if D[row, col] > self.maxDistance:
                        if self.visualize == True:
                                print ('CentroidTracker: ID {} not matched to rectangle {} because D={} > maxDistance = {}'.format(objectIDs[row], inputRectangles[col], D[row,col], self.maxDistance))
                        continue

                    # otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    # counter

                    objectID = list(self.objects.keys())[objectIDs[row]-accumulatedDisappeareance]
                    if self.visualize == True:
                        print("We have matched the new object with category "+ str(category)+" with centroids which were "+str(self.objects[objectID])+ " to "+ str(inputCentroids[col]))
                    self.objects[objectID]     = inputCentroids[col]
                    self.rectangles[objectID]  = inputRectangles[col] # JRMR
                    self.disappeared[objectID] = 0
                    
                    # indicate that we have examined each of the row and
                    # column indexes, respectively
                    usedRows.add(row)
                    usedCols.add(col)
            
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)


            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    if self.visualize == True:
                            print ('CentroidTracker: ID {} marked as disappeared'.format(objectIDs[row]))

                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = list(self.objects.keys())[objectIDs[row]-accumulatedDisappeareance]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        if self.visualize == True:
                                print ('Deregistering ID {} after {} frames'.format(objectID, self.maxDisappeared))
                                sys.stdout.flush()
                        self.deregister(objectID)
                        accumulatedDisappeareance +=1
                 

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    if self.visualize == True:
                        print("We have registered the new input Object with category "+ str(category)+" found in position "+ str(col) +" with inputRectangles "+ str(inputCentroids[col]))
                    self.register(inputCentroids[col], category, inputRectangles[col])   # JRMR: added argument inputRectangles[col]  EVM: added argument categories[col]
            return accumulatedDisappeareance
                                                          