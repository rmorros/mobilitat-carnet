from multiprocessing import Process
import sys
import cv2
import os
import imutils
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image
from multiprocessing import Process
from multiprocessing import Queue
from torchvision.models import vgg16


rocket = 0


total=0
bikebi=0
bikeu=0
road=0
sidewalk=0
BlockDuration = int(sys.argv[1])
MAXBlockDuration = int(sys.argv[2])

result_list=["unknown", 0]


def image_loader(image_name):#Single image loader

    """load image, returns cuda tensor"""
    image = image_name
    image = transform(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

if not torch.cuda.is_available():
    raise Exception("You should enable GPU in the runtime menu.")
device = torch.device("cuda:0")

imgsize = 150

transform = transforms.Compose(
    [transforms.Resize(int(imgsize)),  # Resize the short side of the image to 150 keeping aspect ratio
     transforms.CenterCrop(int(imgsize)),  # Crop a square in the center of the image
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('BikeBi', 'BikeU', 'Crosswalk', 'Road', 'Sidewalk')

'''
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(0.5),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(0.5),
    nn.Flatten(),
    nn.Linear(int((imgsize / 4)) * int((imgsize / 4)) * 64, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4),
    nn.LogSoftmax(dim=1)
)
model.to(device)
'''


pretrained_model = vgg16(pretrained=True)
pretrained_model.eval()
pretrained_model.to(device)

feature_extractor = pretrained_model.features

for layer in feature_extractor[:24]:  # Freeze layers 0 to 23
    for param in layer.parameters():
        param.requires_grad = False

for layer in feature_extractor[24:]:  # Train layers 24 to 30
    for param in layer.parameters():
        param.requires_grad = True


model = nn.Sequential(
        feature_extractor,
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(int((imgsize/37.5))*int((imgsize/37.5))* 512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 5),
        nn.Softmax(dim=1)
)

model.to(device)


MODEL_PATH = ".\Models\model.pth"
model.load_state_dict(torch.load(MODEL_PATH))


i=0
timestamp=0

result_list=["unknown", 0]

x=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
y=[0.974151057489855, 0.975350607979488, 0.9713539146424078, 0.9749905377471104, 0.9774039922991185, 0.9764190960974858, 0.9784793203087937, 0.9722327881323697, 0.9738625110752072, 0.9784193144288805, 0.9758846900120115, 0.9725752288141173, 0.9750078756694319, 0.9746887044174326, 0.9762444322888177, 0.9769110892388452, 0.9749344339825444, 0.9803446418955304, 0.9805363117960318, 0.9776047209054851]
name = " "
name_aux = " "
while(BlockDuration<=MAXBlockDuration):
    file = open('./PythonCode/notes.txt', 'r') 
    i=0
    timestamp=0
    total=0
    bikebi=0
    bikeu=0
    road=0
    sidewalk=0
    crosswalk=0
    predicted_block="unknown"
    corrects=0
    incorrects=0
    for line in file:
        k=-1
        for word in line.split():
            k+=1
            if k == 0:
                if(name!=word):
                    name = word 
                    print("----------------- Starting analysis of " + name + ".mp4 -----------------") 
                else:
                    name = word 
            if k == 1:
                T0str = word
            if k == 2:
                T1str = word
            if k == 3:
                Type = word 
                '''
                print("Name: " + name) 
                print("T0: " + T0str) 
                print("T1: " + T1str) 
                print("Type: " + Type) 
                '''

        #converts time input to seconds (float)
        m, s, f = T0str.split('.')
        T0= float(s) + float(m)*60 + float(f)/1000
        m, s, f = T1str.split('.')
        T1= float(s) + float(m)*60 + float(f)/1000

        video2 = cv2.VideoCapture(f"./images/{name}.mp4")
        fps = video2.get(cv2.CAP_PROP_FPS)
        ms = int(1000/(fps))

        #Sets the counter to T0 frame, to start analyzing the clip at that time
        video2.set(1,int(T0*fps))
        i=int(T0*fps)
        while (video2.isOpened()):
            timestamp=i/fps
            #Read the video
            ret, orig_frame = video2.read()
            if ret:
                #Transform the image to input it to the model
                #orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
                orig_frame_PIL = Image.fromarray(orig_frame)
                image = image_loader(orig_frame_PIL)
                outputs=model(image)
                '''
                #Prediction
                _, predicted = torch.max(outputs, 1)
                #Stats calculation
                if classes[predicted[0]] == 'BikeU':
                    bikeu+=1
                if classes[predicted[0]] == 'BikeBi':
                    bikebi+=1
                if classes[predicted[0]] == 'Sidewalk':
                    sidewalk+=1
                if classes[predicted[0]] == 'Road':
                    road+=1
                '''
                bikebi+=outputs[0][0]
                bikeu+=outputs[0][1]
                crosswalk+=outputs[0][2]
                road+=outputs[0][3]
                sidewalk+=outputs[0][4]

                i+=1
                
            if (i%BlockDuration==0) | (not ret) | (i>=int(T1*fps)):  
                total=bikebi+crosswalk+bikeu+road+sidewalk
                '''
                print(f"Probability of BikeBi: {round(float(bikebi/total)*100, 3)}%")
                print(f"Probability of BikeU: {round(float(bikeu/total)*100, 3)}%")
                print(f"Probability of Crosswalk: {round(float(crosswalk/total)*100, 3)}%")
                print(f"Probability of Road: {round(float(road/total)*100, 3)}%")
                print(f"Probability of Sidewalk: {round(float(sidewalk/total*100), 3)}%")
                '''
                outputs=torch.tensor([[bikebi, bikeu, crosswalk, sidewalk, road]], device='cuda:0')

                _, predicted = torch.max(outputs, 1)

                if bikeu > bikebi and bikeu > sidewalk and bikeu > road and bikeu>crosswalk:
                    predicted_block="BikeU"
                elif bikebi > bikeu and bikebi > sidewalk and bikebi > road and bikebi>crosswalk:
                    predicted_block="BikeBi"
                elif sidewalk > bikeu and sidewalk > bikebi and sidewalk > road and sidewalk > crosswalk:
                    predicted_block="Sidewalk"
                elif road > bikeu and road > sidewalk and road > bikebi and road > crosswalk:
                    predicted_block="Road"
                elif crosswalk > bikeu and crosswalk > sidewalk and crosswalk > bikebi and crosswalk > road:
                    predicted_block="Crosswalk"
                else:
                    predicted_block="unknown"

                #Print the predicted class
                '''
                print(f'Predicted {BlockDuration-i%BlockDuration} frames BLOCK (AT FRAME {i}) class: {predicted_block}')
                print("")
                '''
                if(predicted_block==Type):
                    aux_size = i%BlockDuration
                    if aux_size==0:
                        aux_size=60
                    corrects += aux_size
                else:
                    aux_size = i%BlockDuration
                    if aux_size==0:
                        aux_size=60
                    incorrects += aux_size
                if(name!=name_aux):
                    print(f"Current precision: {round(float(corrects/(corrects+incorrects))*100, 2)}%")
                    name_aux=name

                timestamp_offset=timestamp
                bikebi=0
                bikeu=0
                sidewalk=0
                crosswalk=0
                road=0

            if i>=int(T1*fps):
                break
            if not ret:
                break


    timestamp=result_list[1]
    timestamp_offset=result_list[1]
    '''
    print("Predictions Finished")
    print("")
    '''
    print(f"Block Size: {BlockDuration} // Correct: {round(float(corrects/(corrects+incorrects))*100, 2)}%")
    print("")
    x.append(BlockDuration)
    print(x)
    y.append((corrects/(corrects+incorrects)))
    print(y)
    print("")
    BlockDuration += 5

# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('Block Duration') 
# naming the y axis 
plt.ylabel('Precision') 
  
# giving a title to my graph 
plt.title('Statistics') 
  
plt.show()

# function to save the plot 
plt.savefig(f".\Results\Block_sizes.jpg")
