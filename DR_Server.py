#!/usr/bin/env python
# coding: utf-8

# # Dick Rater FastAPI- Server

# ### Import libraries

# In[1]:


import cv2
import numpy as np
from ultralyticsplus import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

import io
import uvicorn
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import os

from concurrent.futures import ThreadPoolExecutor


# # Calibrator and Dick Detector

# In[2]:


class Calibrator():
    
    def __init__(self,circle_r_cm):
        self.max_image_dim=800
        self.circle_r_cm=circle_r_cm
        self.ratio=1
            
    def resize_image(self,img):
        
        height, width, channels = img.shape
        max_dim = max(width, height)
        self.ratio=1
        if max_dim > self.max_image_dim:
            self.ratio = self.max_image_dim / max_dim
            new_size = (int(width * self.ratio), int(height * self.ratio))
            img = cv2.resize(img, new_size)
        
        return img
    
    def preprocess_image(self,image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT) 
        gray = cv2.Canny(gray, 50, 100)
        gray = cv2.dilate(gray,None, iterations=4)
        
        return gray

    def detect_and_measure_circle_r(self,gray,image):
        # detect circles in the image
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                       param1=100, param2=30,
                                       minRadius=1, maxRadius=200)
        # ensure at least some circles were found
        if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                max_mean=-10000000000000
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                        # Check if the area inside the rectangle is white
                        x1=x-r
                        x2=x+r
                        y1=y-r
                        y2=y+r
                        rect_img = image[y1:y2, x1:x2,:]
                        rect_mean = rect_img.mean()
                       
                        if(rect_mean>max_mean):
                                max_mean=rect_mean
                                coca_cola_circle_xyr=(x, y, r)          
                return coca_cola_circle_xyr
        else:
                return 0
    
    def draw_detected_circle_on_image(self,img):
        
        resized_image=self.resize_image(img)
        rs=cv2.cvtColor(self.resize_image(img), cv2.COLOR_BGR2RGB)
        preprocessed_image=self.preprocess_image(resized_image)
        circle_xyr=self.detect_and_measure_circle_r(preprocessed_image,resized_image)  
        if(circle_xyr!=0):
            drawn_circle=cv2.circle(resized_image, (circle_xyr[0], circle_xyr[1]),circle_xyr[2], (0, 0, 255), 4)
            drawn_circle = cv2.cvtColor(drawn_circle, cv2.COLOR_BGR2RGB)


            gr=cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
            image_couples=np.hstack([rs,drawn_circle,gr])
            return image_couples
        else:
            gr=cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
            image_couples=np.hstack([rs,gr])
            return image_couples
            
    def calculate_ppcm(self,img):
        
        resized_image=self.resize_image(img)
        preprocessed_image=self.preprocess_image(resized_image)
        circle_xyr=self.detect_and_measure_circle_r(preprocessed_image,resized_image)
        if(circle_xyr!=0):
            circle_r=circle_xyr[2]
            ppcm=(circle_r/self.ratio)/self.circle_r_cm

            return ppcm
        else:
            return 0


# In[3]:


"""
Dick detector can detect abnormal cases along with normal case:

Theoretically there can be 9 cases : 8 abnormal and 1 normal.

The normal case:
0) Normal case is ONE dick ONE dick-head 

Abnormal cases:
0)No dick No dick-head 
1)Dick and dick head are detected both but distance between them is very large. 
  SOLUTION-Drop dick-head and take dick. Take dick's bbw as dick width.
  
2)Multiple dicks NO dick-head- Error,No solution
3)Multiple dick-heads NO dick -Error,No solution

4)One dick NO dick head -SOLUTION : Dick's bbw is set as dick width.
5)One dick-head NO dick - Error,No solution

6)Multiple dick heads Multiple dicks- Error,No solution

7)One dick Multiple dick heads-SOLUTION: Dick-head closest to dick will be chosen and other will be ignored

8) if dick width and length proportions are bad or length and width values are unreal circle detection is wrong
"""


class Dick_Detector:

    def __init__(self,model_path):
        
         
        self.model = self.load_model(model_path)
        self.max_dist=25# cm
        self.rot_angle=360
        self.rot_angle_step=15
        self.rot_angle_step_finer=4
        self.CLASS_NAMES_DICT = self.model.model.names    
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
        self.max_image_dim=2300
        self.ratio=1
            
    def resize_image(self,img):
        
        height, width, channels = img.shape
        max_dim = max(width, height)
        
        if max_dim > self.max_image_dim:
            self.ratio = self.max_image_dim / max_dim
            new_size = (int(width * self.ratio), int(height * self.ratio))
            img = cv2.resize(img, new_size)
        return img   



    def load_model(self,model_path):
       
        model = YOLO(model_path)  # load a pretrained YOLOv8 model
        model.fuse()
    
        return model
    
    def set_model_params(self,conf=0.20,iou=0.45,agnostic_nms=False,max_det=1000,augment=True):
        self.model.overrides['conf'] = conf  # NMS confidence threshold
        self.model.overrides['iou'] = iou  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
        self.model.overrides['max_det'] = max_det  # maximum number of detections per image    
        self.model.overrides['augment'] = augment  # augmentation during inference   


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def calculate_euclidean_distance(self,ls1,ls2):
        return np.sqrt((ls1[0]-ls2[0])**2+(ls1[1]-ls2[1])**2)
    


    def rotate_image(self,mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        if(angle<0):
            angle=360+angle
        mat=self.resize_image(mat)    
        height, width = mat.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        
        
        return rotated_mat    

    def find_rot_angle(self,image):    
        # Rotate the image using multiple threads

        #1)#Take input image and Generate 36 rotated images- 360-degree, 10 degree step
        angles=[int(i) for i in range(0,self.rot_angle,self.rot_angle_step)]
        with ThreadPoolExecutor() as executor:
            rotated_images = list(executor.map(lambda angle: self.rotate_image(image, angle), angles))
        
        #2)Make prediction on 36 images and find in which image ratio of dy/dx is highest. Call it "max ratio image"
        r=self.predict(rotated_images) 
        dick_length=0
        dick_width=0
        #Broad Search -10 degree accuracy
        rxy_max=-100000000
        rot_angle=0
        rotated_image=image
        results_0=None
        z=0
        for k in enumerate(r):
            for j,result in enumerate(k[1]):
                class_id = result.boxes.cls.cpu().numpy().astype(int)
                if class_id == 0:
                    xyxy=result.boxes.xyxy.cpu().numpy()                
                    dx=abs(xyxy[0][0]-xyxy[0][2])
                    dy=abs(xyxy[0][1]-xyxy[0][3])
                    rxy=dy/dx
                    if(rxy>rxy_max):
                        rxy_max=rxy
                        rot_angle=angles[z]
                        dick_length=dy
                        dick_width=dx
                        rotated_image=rotated_images[z]
                        results_0=k[1]
            z=z+1

        #3) #Take "max ratio image"  and Generate 20 rotated images 20 degree, 1 degree step. 
        angles=[int(i) for i in range(rot_angle-self.rot_angle_step+1,rot_angle+self.rot_angle_step+1,self.rot_angle_step_finer)]
        with ThreadPoolExecutor() as executor:
            rotated_images = list(executor.map(lambda angle: self.rotate_image(rotated_image, angle), angles))                

        #4)Find in which image ratio of dy/dx is highest. Call it "maxmax ratio image"  
        r=self.predict(rotated_images) 

        #Broad Search -10 degree accuracy
        rxy_max=rxy_max
        z=0
        for k in enumerate(r):
            for j,result in enumerate(k[1]):
                class_id = result.boxes.cls.cpu().numpy().astype(int)
                if class_id == 0:
                    xyxy=result.boxes.xyxy.cpu().numpy()                
                    dx=abs(xyxy[0][0]-xyxy[0][2])
                    dy=abs(xyxy[0][1]-xyxy[0][3])
                    rxy=dy/dx
                    if(rxy>rxy_max):
                        rxy_max=rxy
                        rot_angle=angles[z]
                        dick_length=dy
                        dick_width=dx
                        rotated_image=rotated_images[z]  
                        results_0=k[1]
            z=z+1

        return rotated_image,dick_length,dick_width,results_0
        
           

    def calculate_length_and_width(self, results,image,ppcm):
        
        rotated_image,dick_length,dick_width,_=self.find_rot_angle(image)
#         rotated_image=self.rotate_image(image,rot_angle)   
        results=self.predict(rotated_image)
        
        """ Extract Dicks and Dick-Heads indicies,bounding boxes and results"""
        dick_length=0 
        dick_width=0        
        dicks=[]
        dick_heads=[]
        class_ids=[]
        # Extract detections
        for i,result in enumerate(results[0]):
            #Extract and store class id
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            # Artificially excluding some detetions to create different abnormal cases
#             print(result.boxes.conf.cpu().numpy())
#             if(result.boxes.conf.cpu().numpy()<0.55 or result.boxes.conf.cpu().numpy()>0.75):
#                 class_ids.append(class_id)
            class_ids.append(class_id)
            #If dick is detected
            if class_id == 0:
                #calclate bounding box length and width
                xyxy=result.boxes.xyxy.cpu().numpy()                
                #Store dick index,bounding box centers,and results
                dick_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2]
                dicks.append((i,dick_bb_center,result))                        
            #If dick head is detected
            else:
                #calculate bounding box length and width
                xyxy=result.boxes.xyxy.cpu().numpy()
                #Store dick-head index,bounding box centers,and results
                dick_head_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2]    
                dick_heads.append((i,dick_head_bb_center,result))
        

        
        """HANDLING ABNORMAL CASE 0"""
        STATUS_CODE="AC0"
        if(len(class_ids)==0):
            print("ABNORMAL CASE 0")
            STATUS_CODE="AC0"
            rotated_image,dick_length,dick_width,_=self.find_rot_angle(image)    
            
            
        """HANDLING NORMAL CASE AND ABNORMAL CASE 1"""
        #If 1 dick and 1 dick-head are detected but distance between them is larger than threshold self.max_dist
        if(len(class_ids)==2 and (0 in class_ids) and (1 in class_ids) ):
            dick_bb_center=dicks[0][1]
            dick_head_bb_center=dick_heads[0][1]
            dist=self.calculate_euclidean_distance(dick_bb_center,dick_head_bb_center)
            #Calibrate distance
            calibrated_dist=dist/ppcm;
            print("dist=",calibrated_dist)
            if(calibrated_dist>self.max_dist):
                """HANDLING ABNORMAL CASE"""
                print("ABNORMAL CASE 1")
                STATUS_CODE="AC1"
                #SOLUTION-Drop dick-head and take dick. Take dick's bbw as dick width.
                #print("rot_angle=",rot_angle)
                rotated_image,dick_length,dick_width,_=self.find_rot_angle(image)     
            else:
                
                """HANDLING NORMAL CASE"""
                print("NORMAL CASE")
                STATUS_CODE="NC"
                rotated_image,dick_length,dick_width,_=self.find_rot_angle(image)                
                       
        
        """HANDLING ABNORMAL CASE 2"""                        
        if(len(class_ids)>=2 and (0 in class_ids) and not (1 in class_ids)):
            print("ABNORMAL CASE 2")
            STATUS_CODE="AC2"
        
        
        """HANDLING ABNORMAL CASE 3"""    
        if((1 in class_ids) and not (0 in class_ids)):
            print("ABNORMAL CASE 3")
            STATUS_CODE="AC3"
            
            
        """HANDLING ABNORMAL CASE 4"""   
        if(len(class_ids)==1 and (0 in class_ids)):
            print("ABNORMAL CASE 4")
            STATUS_CODE="AC4"
            rotated_image,dick_length,dick_width,_=self.find_rot_angle(image)
            #print("rot_angle=",rot_angle)
            
        """HANDLING ABNORMAL CASE 5"""   
        if((len(class_ids)==1 and (1 in class_ids)) or (STATUS_CODE=="AC0")):
            print("ABNORMAL CASE 5")
            STATUS_CODE="AC5"
            class_ids=[]
            rotated_image,dick_length,dick_width,results=self.find_rot_angle(image)
#             results_0=self.predict(rotated_image)
            dicks=[]
            dick_heads=[]
            class_ids=[]
            results=results[1]
            
            # Extract detections
            for i,result in enumerate(results[0]):
                #Extract and store class id
                class_id = result.boxes.cls.cpu().numpy().astype(int)
                class_ids.append(class_id)
                #If dick is detected
                if class_id == 0:
                    #calclate bounding box length and width
                    xyxy=result.boxes.xyxy.cpu().numpy()                
                    #Store dick index,bounding box centers,and results
                    dick_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2]
                    dicks.append((i,dick_bb_center,result))                        
                #If dick head is detected
                else:
                    #calculate bounding box length and width
                    xyxy=result.boxes.xyxy.cpu().numpy()
                    #Store dick-head index,bounding box centers,and results
                    dick_head_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2]    
                    dick_heads.append((i,dick_head_bb_center,result))
            
########################################################################################################            
            """HANDLING NORMAL CASE AND ABNORMAL CASE 1"""
            #If 1 dick and 1 dick-head are detected but distance between them is larger than threshold self.max_dist
            if(len(class_ids)==2 and (0 in class_ids) and (1 in class_ids) ):
                dick_bb_center=dicks[0][1]
                dick_head_bb_center=dick_heads[0][1]
                dist=self.calculate_euclidean_distance(dick_bb_center,dick_head_bb_center)
                #Calibrate distance
                calibrated_dist=dist/ppcm;
                #print("dist=",calibrated_dist)
                if(calibrated_dist>self.max_dist):
                    """HANDLING ABNORMAL CASE"""
                    STATUS_CODE="AC1"
                else:

                    """HANDLING NORMAL CASE"""
                    print("NORMAL CASE")
                    STATUS_CODE="NC"
                    

            """HANDLING ABNORMAL CASE 2"""                        
            if(len(class_ids)>=2 and (0 in class_ids) and not (1 in class_ids)):
                print("ABNORMAL CASE 2")
                STATUS_CODE="AC2"


            """HANDLING ABNORMAL CASE 3"""    
            if((1 in class_ids) and not (0 in class_ids)):
                print("ABNORMAL CASE 3")
                STATUS_CODE="AC3"


            """HANDLING ABNORMAL CASE 4"""   
            if(len(class_ids)==1 and (0 in class_ids)):
                print("ABNORMAL CASE 4")
                STATUS_CODE="AC4"
                #print("rot_angle=",rot_angle)            

            """HANDLING ABNORMAL CASE 6"""                        
            if(len(class_ids)>3 and  class_ids.count(0)>1 and class_ids.count(1)>1):
                print("ABNORMAL CASE 6")
                STATUS_CODE="AC6"

            """HANDLING ABNORMAL CASE 7"""    
            if(len(class_ids)>2 and class_ids.count(0)==1):
                print("ABNORMAL CASE 7")
                STATUS_CODE="AC7"
                #Rotate image and make predictions
                rotated_image,dick_length,dick_width,results=self.find_rot_angle(image)
                #print("rot_angle=",rot_angle)
#                 results=self.predict(rotated_image)
                results=results[1]
                #Extract detections and find closest dick-head to the dick
                min_dist=1000000000
                for i,result in enumerate(results[0]):
                    class_id = result.boxes.cls.cpu().numpy().astype(int)
                    if class_id == 0:
                        xyxy=result.boxes.xyxy.cpu().numpy()                
                        dick_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2]
                        dy=abs(xyxy[0][1]-xyxy[0][3])
                        """Set dick length"""
                        dick_length=dy
                    else:
                        xyxy=result.boxes.xyxy.cpu().numpy()
                        dick_head_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2] 
                        dx=abs(xyxy[0][0]-xyxy[0][2])
                        #Compare distance
                        dist=self.calculate_euclidean_distance(dick_bb_center,dick_head_bb_center)
                        """Set dick width"""
                        if(dist<min_dist):
                            min_dist=dist
                            dick_width=dx                
                
                
                
########################################################################################################################            

        """HANDLING ABNORMAL CASE 6"""                        
        if(len(class_ids)>3 and  class_ids.count(0)>1 and class_ids.count(1)>1):
            #print("ABNORMAL CASE 6")
            STATUS_CODE="AC6"
            
        """HANDLING ABNORMAL CASE 7"""    
        if(len(class_ids)>2 and class_ids.count(0)==1):
            print("ABNORMAL CASE 7")
            STATUS_CODE="AC7"
            #Rotate image and make predictions
            rotated_image,dd,bb,results=self.find_rot_angle(image)
#             results=self.predict(rotated_image)
            results=results[1]
            #Extract detections and find closest dick-head to the dick
            min_dist=1000000000
            for i,result in enumerate(results[0]):
                class_id = result.boxes.cls.cpu().numpy().astype(int)
                if class_id == 0:
                    xyxy=result.boxes.xyxy.cpu().numpy()                
                    dick_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2]
                    dy=abs(xyxy[0][1]-xyxy[0][3])
                    """Set dick length"""
                    dick_length=dy
                else:
                    xyxy=result.boxes.xyxy.cpu().numpy()
                    dick_head_bb_center=[(xyxy[0][0]+xyxy[0][2])/2,(xyxy[0][1]+xyxy[0][3])/2] 
                    dx=abs(xyxy[0][0]-xyxy[0][2])
                    #Compare distance
                    dist=self.calculate_euclidean_distance(dick_bb_center,dick_head_bb_center)
                    """Set dick width"""
                    if(dist<min_dist):
                        min_dist=dist
                        dick_width=dx
                                    
        calibrated_dick_length=dick_length/(ppcm*self.ratio)
        calibrated_dick_width=dick_width/(ppcm*self.ratio)
        print("calibrated_dick_length=",calibrated_dick_length)
        print("calibrated_dick_width=",calibrated_dick_width)
        return calibrated_dick_length,calibrated_dick_width,STATUS_CODE
    
    def predict_on_image(self,image,ppcm):
        results = self.predict(image)
        dick_length,dick_width,STATUS_CODE=self.calculate_length_and_width(results,image,ppcm)
        return dick_length,dick_width,STATUS_CODE
     
    
    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
                
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
        # Setup detections for visualization
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)
        
        return frame
    
    def predict_on_image_and_visualize(self,image):
        results = self.predict(image)
        image = self.plot_bboxes(results, image)
        return image


# # Dick Rater

# In[4]:


class Dick_Rater():
    
    def __init__(self,model_path,circle_r_cm):
        
        self.C=Calibrator(circle_r_cm) # #Instatiate calibrator
        self.DD=Dick_Detector(model_path) # Instatiate dick detector
        self.DD.set_model_params() # Set dick detector params
        self.DISALLOWED_STATUS_CODES=["AC0","AC2","AC3","AC5","AC6"]
        self.ALLOWED_STATUS_CODES=["NC","AC1","AC4","AC7"]
        self.min_dick_length=5
   
    
    def rate_dick(self,image):


        
        #Calculate ppcm
        ppcm=self.C.calculate_ppcm(image)
        #If circle was found ppcm is not equal to 0
        if(ppcm==0):
            dick_data= "Error: Circle not found"
        else:
            #Detect dick
            calibrated_dick_length,calibrated_dick_width,STATUS_CODE=self.DD.predict_on_image(image,ppcm)
            #If dick can be detected and measured
            if(STATUS_CODE in self.DISALLOWED_STATUS_CODES):
                dick_data= "Error : {}".format(STATUS_CODE)
            else:
                
                ratio=(calibrated_dick_length/calibrated_dick_width)
                
                if(calibrated_dick_length<self.min_dick_length):
                    dick_data= "Error: Incorrect circle "
                
                elif(ratio<1.5):
                    dick_data= "Error : Bad image "
                    
                else:                    
                    dick_data= [calibrated_dick_length,calibrated_dick_width,STATUS_CODE]
                
                
                
                
        if(type(dick_data)==str):    
            result_type="Error"
            # Create a dictionary with keys "length", "width", "name"
            dick_data_dict = {"result_type":result_type,"message":dick_data}
            return dick_data_dict
        else:
            result_type="Success"
            # Create a dictionary with keys "length", "width", "name"
            rounded_dick_length=round(dick_data[0], 2)
            rounded_dick_width=round(dick_data[1], 2)
            dick_data_dict = {"result_type":result_type,"dick_length": rounded_dick_length, "dick_width": rounded_dick_width, "STATUS_CODE": dick_data[2]}
            return dick_data_dict


# # Configure Dick Rater Model

# In[5]:


#Instatiate Dick Rater
model_path='weights/best_m.pt'# dick detector yolov8
circle_r_cm=1.5 # circle radius
DR=Dick_Rater(model_path,circle_r_cm)


# # Configure FastAPI

# In[6]:


import matplotlib.pyplot as plt
app = FastAPI(title='Dick Rater FastAPI-Server')
# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global k
    contents = await file.read()

    # Convert the bytes to an OpenCV image
    img_np = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dick_data_dict=DR.rate_dick(image)
    
    return dick_data_dict


# In[7]:


# Spin up the uvicorn server

# Allows the server to be run in this interactive environment
nest_asyncio.apply()

# Host depends on the setup you selected (docker or virtual env)
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

# Spin up the server!    
uvicorn.run(app, host=host, port=8000)
