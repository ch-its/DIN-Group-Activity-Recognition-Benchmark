import cv2
import numpy as np
import os
from os.path import isfile, join
import moviepy.video.io.ImageSequenceClip
from config import *
from collections import Counter
import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
import numpy as np
"""

 

cv2.imshow("image",image)
gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Changed",gray)
ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours:" + str(len(contours)))
x,y,w,h = cv2.boundingRect(contours[0])
cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
cv2.imshow("result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def ann(seqno,filepath):
    
    seq = str(seqno)
    ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking']
    ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking']


    ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
    ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}
    Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
    Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}
    class_cnt = [0,0,0,0]
    p = { "0":"Walking" ,"1": "Waiting", "2": "Queueing" , "3":"Crossing"}
    
    
        
        
    
    
    
    annotations = {}
    
    
        
        
    sid = seqno
    if seqno<=9 :
        seq="0"+str(seqno)
    else:
        seq = str(seqno)

    FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
         11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
         21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
         31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
         41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}

    FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
            11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
            21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
            31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
            41: 707, 42: 420, 43: 410, 44: 356}


    fno = str(i)
    if i<=9:
        fno = "000" + fno
    if i>9 and i<=99:
        fno = "00" + fno
    if i > 99 and i<=999:
        fno = "0" + fno
        
            
        
    #print("data/collective_activity_dataset/"+"seq"+str(seq)+"/"+"frame"+str(fno)+".jpg")

    path= "data/collective_activity_dataset/"+"seq"+seq + "/annotations.txt"



    with open(path,mode='r') as f:
        frame_id=None
        group_activity=None
        actions=[]
        bboxes=[]
        for l in f.readlines():


            values=l[:-1].split('	')

            if int(values[0])!=frame_id:
                if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                    counter = Counter(actions).most_common(2)
                    group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                    class_cnt[Activity5to4[group_activity]]+=1
                    annotations[frame_id]={
                        'frame_id':frame_id,
                        'group_activity':group_activity,
                        'actions':actions,
                        'bboxes':bboxes
                    }

                frame_id=int(values[0])
                group_activity=None
                actions=[]
                bboxes=[]

            actions.append(int(values[5])-1)
            x,y,w,h = (int(values[i])  for i  in range(1,5))
            H,W=FRAMES_SIZE[sid]

            bboxes.append( (y/H,x/W,(y+h)/H,(x+w)/W) )

        if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
            counter = Counter(actions).most_common(2)
            group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
            class_cnt[Activity5to4[group_activity]]+=1
            annotations[frame_id]={
                'frame_id':frame_id,
                'group_activity':group_activity,
                'actions':actions,
                'bboxes':bboxes
            }
            


    with open(filepath) as file_out:
        lines_out = []
        for line_out in file_out:
            lines_out.append(line_out.split(","))
    #sprint(lines_out)


    

    


    ans_cnt = 0

    #print(len(d_out))



    

    height = 0
    width = 0
    if not os.path.isdir("result_din_cad_res"):
        os.makedirs("result_din_cad_res")
    print("length of d",len(d))
    print("length of lines_out",len(lines_out))
    for i in d:
        
        imgpath = "data/collective_activity_dataset/"+"seq"+str(seqno)+"/"+"frame"+str(i)+".jpg" 
        image = cv2.imread(imgpath)
        

        print(d[i])
        
        bboxes = d[i]["bboxes"]
        for j in bboxes:
            
            X = int(j[1])
            Y = int(j[0])
            W = int(j[3])-X
            H = int(j[2])-Y

            height = H
            width = W
            image = cv2.rectangle(image,(X,Y),(X+W,Y+H),(255,0,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (37, 70)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (0, 255, 0)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(image, "Actual Value", org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        org = (37, 100)
        #print(d_out[ans_cnt][0][0])
        image = cv2.putText(image, p[int(lines_out[ans_cnt][0][0])], org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        # org
        org = (437, 70)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (0, 0, 255)

        image = cv2.putText(image, "Predicted Value", org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        org = (437, 100)

        #print(int(lines_out[ans_cnt][1][0]))
        print("ans_cnt :",ans_cnt)
        image = cv2.putText(image, p[int(lines_out[ans_cnt][1][0])], org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        #print("result_"+"seq"+seq+"/frame" + i + ".jpg")
        cv2.imwrite("result_din_cad_res/"+ str(i) + ".jpg",image)

        ans_cnt+=1








    def convert_frames_to_video(pathIn,pathOut,fps):
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        files.sort(key = lambda x: int(x[5:-4]))
        #print(files)
        #for sorting the file names properly
        for i in range(len(files)):
            filename=pathIn + files[i]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            #print(filename)
            #inserting the frames into an image array
            frame_array.append(img)
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()

    
    image_folder='result_din_vbd4_res'
    fps=1

    image_files = [os.path.join(image_folder,img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".jpg")]
    image_files.sort(key = lambda x: int(x[20:len(x)-4]))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    if not os.path.isdir('result_din__vbd'+"seq"+seq+'_vid__res'):
        os.makedirs('result_din__vbd'+"seq"+seq+'_vid__res')
    clip.write_videofile('result_din__vbd'+"seq"+seq+'_vid__res/output3.mp4')

    
    
    
   
    
    print("done")


def gen_ann():
    
    
  
    # Folder Path
    path = "task_factory videos-2022_05_05_03_22_14-yolo 1.1/obj_train_data/"

    # Change the directory
    os.chdir(path)

    # Read text File


    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            print(f.read())


    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"

            # call read text file function
            read_text_file(file_path)



def count_ann():
    
    annotations={}
    
    ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking']
    ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking']


    ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
    ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}
    Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
    Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}
    class_cnt = [0,0,0,0]
    p = { "0":"Walking" ,"1": "Waiting", "2": "Queueing" , "3":"Crossing"}
    seq = ""
    
    test_seqs = [5,6,7,8,9,10,11,15,16,25,28,29]
    train_seqs = [s for s in range(1,45) if s not in test_seqs]
    
    for seqno in test_seqs:
        
        
        sid = seqno
        if seqno<=9 :
            seq="0"+str(seqno)
        else:
            seq = str(seqno)
        
        FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
             11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
             21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
             31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
             41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}
        
        FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
                11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
                21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
                31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
                41: 707, 42: 420, 43: 410, 44: 356}
    
    
        path= "data/collective_activity_dataset/"+"seq"+seq + "/annotations.txt"
         
    
    
        with open(path,mode='r') as f:
            frame_id=None
            group_activity=None
            actions=[]
            bboxes=[]
            for l in f.readlines():


                values=l[:-1].split('	')

                if int(values[0])!=frame_id:
                    if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                        counter = Counter(actions).most_common(2)
                        group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                        class_cnt[Activity5to4[group_activity]]+=1
                        annotations[frame_id]={
                            'frame_id':frame_id,
                            'group_activity':group_activity,
                            'actions':actions,
                            'bboxes':bboxes
                        }

                    frame_id=int(values[0])
                    group_activity=None
                    actions=[]
                    bboxes=[]

                actions.append(int(values[5])-1)
                x,y,w,h = (int(values[i])  for i  in range(1,5))
                H,W=FRAMES_SIZE[sid]

                bboxes.append( (y/H,x/W,(y+h)/H,(x+w)/W) )

            if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                counter = Counter(actions).most_common(2)
                group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                class_cnt[Activity5to4[group_activity]]+=1
                annotations[frame_id]={
                    'frame_id':frame_id,
                    'group_activity':group_activity,
                    'actions':actions,
                    'bboxes':bboxes
                }
        
            
                
         


    


        
          

    for i in range(len(class_cnt)):
        print("Count of "+p[str(i)]+" is ",class_cnt[i])

        




    #print(len(d_out))



        

        







"""





"""

# ann(3)
count_ann()
# gen_ann()
# ann(4)
