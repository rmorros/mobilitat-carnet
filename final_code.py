from multiprocessing import Process
import sys
import cv2
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
import datetime


VIDEO_PATH = str(sys.argv[1])
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
ms = int(1000/fps)
BlockDuration = int(sys.argv[2])

result_list=["unknown", 0, 0]


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
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.2245, 0.225))])

classes = ('BikeBi', 'BikeU', 'Crosswalk', 'Road', 'Sidewalk')

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

def func1(queue):
    #Loop for every video frame 
    filename=str(datetime.datetime.now())
    filename=filename.replace(":", "-")
    results_file= open(f"./LOGS/results-{filename}.txt","w+")
    results_file.write("0:00:00.000               ")
    results=queue.get()
    TITLE, frame, percentage = results[0], results[1], results[2]
    print(f"{frame}")
    j=0
    h=0
    aux_TITLE = " "
    while True:
        ret, orig_frame = video.read()
        if not ret:
            break
            
        if j==frame:
            h+=1
            results=queue.get()
            TITLE, frame = results[0], results[1]
            if aux_TITLE != TITLE:
                if aux_TITLE != " ":
                    percentage = percentage/h
                    timestamp1 = str(datetime.timedelta(seconds=round(float(j/fps), 3)))
                    if "." in timestamp1:
                        timestamp1 = timestamp1[:-3]
                    else:
                        timestamp1+=str(".000")
                    results_file.write(f"{timestamp1}               {aux_TITLE}\t\t{round(percentage, 2)}%\n{timestamp1}               ")
                    percentage=0
                    h=0
                aux_TITLE=TITLE
            percentage+=results[2]

        #Reshape the image for display depending on its aspect ratio
        if orig_frame.shape[0] > orig_frame.shape[1]:
            orig_frame = imutils.resize(orig_frame, height=900)
        else:
            orig_frame = imutils.resize(orig_frame, width=900)

        #Text and box generator
        cv2.rectangle(orig_frame, (35,15), (195,65), (0,0,0), cv2.FILLED)
        cv2.rectangle(orig_frame, (40,20), (190,60), (255,255,255), cv2.FILLED)
        cv2.putText(orig_frame,  TITLE,  (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1,  (255, 0, 0), 2,  cv2.LINE_8) 

        #Display images
        cv2.imshow("frame", orig_frame)

        #Wait the corresponding ms (to maintain original FPS) and exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        j+=1
    
    video.release()
    cv2.destroyAllWindows()

    h+=1
    percentage = percentage/h
    timestamp1 = str(datetime.timedelta(seconds=round(float(j/fps), 3)))
    if "." in timestamp1:
        timestamp1 = timestamp1[:-3]
    else:
        timestamp1+=str(".000")
    results_file.write(f"{timestamp1}               {aux_TITLE}\t\t{round(percentage, 2)}%")
    results_file.close()
    print ('Video Finished')



def func2(queue):
    i=0
    total=0
    bikebi=0
    bikeu=0
    road=0
    sidewalk=0
    crosswalk=0
    predicted_block="unknown"
    prediction_percentage = 0
    global result_list
    while True:
        #Read the video
        ret, orig_frame = video.read()
        if ret:
            #Transform the image to input it to the model
            #orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            orig_frame_PIL = Image.fromarray(orig_frame)
            image = image_loader(orig_frame_PIL)
            outputs=model(image)
            bikebi+=outputs[0][0]
            bikeu+=outputs[0][1]
            crosswalk+=outputs[0][2]
            road+=outputs[0][3]
            sidewalk+=outputs[0][4]

            i+=1
            
        if (i%BlockDuration==0) | (not ret):  
            total=bikebi+crosswalk+bikeu+road+sidewalk
            print(f"Probability of BikeBi: {round(float(bikebi/total)*100, 3)}%")
            print(f"Probability of BikeU: {round(float(bikeu/total)*100, 3)}%")
            print(f"Probability of Crosswalk: {round(float(crosswalk/total)*100, 3)}%")
            print(f"Probability of Road: {round(float(road/total)*100, 3)}%")
            print(f"Probability of Sidewalk: {round(float(sidewalk/total*100), 3)}%")
            
            outputs=torch.tensor([[bikebi, bikeu, crosswalk, sidewalk, road]], device='cuda:0')

            _, predicted = torch.max(outputs, 1)

            if bikeu > bikebi and bikeu > sidewalk and bikeu > road and bikeu>crosswalk:
                predicted_block="BikeU"
                prediction_percentage = round(float(bikeu/total)*100, 3)
            elif bikebi > bikeu and bikebi > sidewalk and bikebi > road and bikebi>crosswalk:
                predicted_block="BikeBi"
                prediction_percentage = round(float(bikebi/total)*100, 3)
            elif sidewalk > bikeu and sidewalk > bikebi and sidewalk > road and sidewalk > crosswalk:
                predicted_block="Sidewalk"
                prediction_percentage = round(float(sidewalk/total)*100, 3)
            elif road > bikeu and road > sidewalk and road > bikebi and road > crosswalk:
                predicted_block="Road"
                prediction_percentage = round(float(road/total)*100, 3)
            elif crosswalk > bikeu and crosswalk > sidewalk and crosswalk > bikebi and crosswalk > road:
                predicted_block="Crosswalk"
                prediction_percentage = round(float(crosswalk/total)*100, 3)
            else:
                predicted_block="unknown"
                prediction_percentage = 0

            #Print the predicted class
            aux_blocksize = i%BlockDuration
            if aux_blocksize == 0:
                aux_blocksize=BlockDuration
            print(f'Predicted {aux_blocksize} frames BLOCK (AT FRAME {i}) class: {predicted_block}\n\n')
            result_list[0] = predicted_block
            result_list[1] = i
            result_list[2] = prediction_percentage

            queue.put(result_list)
            
            bikebi=0
            bikeu=0
            sidewalk=0
            crosswalk=0
            road=0

        if not ret:
            break

    print("Predictions Finished\n\n")


queue = Queue()

if __name__=='__main__':
    p1 = Process(target=func1, args=(queue,))
    p1.start()
    p2 = Process(target=func2, args=(queue,))
    p2.start()

    p2.join()
    p1.join()

