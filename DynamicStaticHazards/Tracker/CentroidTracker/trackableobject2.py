class TrackableObject:
    def __init__(self, objectID, centroid, startFrame, category, distance = 0):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize a string to set the category
        self.category = category

        self.startFrame = startFrame # Useful to know the span (beginning ~ end) of the object
        self.endFrame = startFrame

        # Contains the distance of the object to the camera
        self.distance = distance

        # Contains some Extra optional Arguments:
            # - If object is Traffic Light: determine the colour of the light : green, orange, red
            # - Annotate if the object is being static or dynamic depending on the velocity computated (default all dynamic)
        self.options = {}
        if self.category == "traffic light":
            self.options["traffic light"] = None
        self.options["static"] = False

    def setOption(self, key, value):
            self.options[key]=value

    def getOption(self, key):
        return self.options[key]

    def __str__(self):
        if self.category == "traffic light":
            return str(self.category) + " with color " + self.options["traffic light"] +" detected on frame " + str(self.startFrame) + " to " + str(self.endFrame) + "\n"
        return str(self.category) + " detected on frame " + str(self.startFrame) + " to " + str(self.endFrame) +"\n"

    def __repr__(self):
        if self.category == "traffic light":
            return str(self.category) + " with color " + self.options["traffic light"] +" detected on frame " + str(self.startFrame) + " to " + str(self.endFrame) + "\n"
        return str(self.category) + " detected on frame " + str(self.startFrame) + " to " + str(self.endFrame) +"\n"
