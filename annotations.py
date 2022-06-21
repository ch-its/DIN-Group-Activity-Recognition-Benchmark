import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
import numpy as np

from collections import Counter

import pickle

from config import *
from gcn_model import *
from base_model import *
from utils import *


import sys
sys.path.append(".")

FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
            11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
            21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
            31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
            41: 707, 42: 420, 43: 410, 44: 356}

 
FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
             11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
             21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
             31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
             41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}


ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking']
ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking']


ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}
Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}


def collective_read_annotations(path,sid):
    annotations={}
    path=path + '/seq%02d/annotations.txt' % sid
    
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
        
        if frame_id!=None  and frame_id<=FRAMES_NUM[sid]:
            counter = Counter(actions).most_common(2)
            group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
            annotations[frame_id]={
                'frame_id':frame_id,
                'group_activity':group_activity,
                'actions':actions,
                'bboxes':bboxes
            }

    return annotations
            
        
        
def collective_read_dataset(path,seqs):
    data = {}
    for sid in seqs:
        data[sid] = collective_read_annotations(path,sid)
    return data

def collective_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s] ]


class CollectiveDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,anns,frames,images_path,image_size,feature_size,num_boxes=13, num_frames = 10, is_training=True,is_finetune=False):
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        
        self.num_boxes = num_boxes
        self.num_frames = num_frames
        
        self.is_training=is_training
        self.is_finetune=is_finetune

        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        # Save frame sequences
        # self.frames_seq[self.flag] = self.frames[index] # [0], self.frames[index][1]
        # if self.flag == 764: # 1336
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/Collective/frames_seq.txt', save_seq)
        # self.flag += 1

        select_frames=self.get_frames(self.frames[index])
        #print("select_frames",select_frames)
        sample=self.load_samples_sequence(select_frames)
        
        return sample
    
    def get_frames(self,frame):
        
        sid, src_fid = frame
        
        if self.is_finetune:
            if self.is_training:
                fid=random.randint(src_fid, src_fid+self.num_frames-1)
                return [(sid, src_fid, fid)]
        
            else:
                return [(sid, src_fid, fid) 
                        for fid in range(src_fid, src_fid+self.num_frames)]
            
        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid,src_fid+self.num_frames),3)
            #     return [(sid, src_fid, fid) for fid in sample_frames]
            #
            # else:
            #     sample_frames=[ src_fid, src_fid+3, src_fid+6, src_fid+1, src_fid+4, src_fid+7, src_fid+2, src_fid+5, src_fid+8 ]
            #     return [(sid, src_fid, fid) for fid in sample_frames]
            if self.is_training:
                return [(sid, src_fid, fid)  for fid in range(src_fid , src_fid + self.num_frames)]
            else:
                return [(sid, src_fid, fid) for fid in range(src_fid, src_fid + self.num_frames)]

    
    
    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        OH, OW=self.feature_size
        
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num=[]
    
        
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg'%(sid,fid))

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)
            
            temp_boxes=[]
            for box in self.anns[sid][src_fid]['bboxes']:
                y1,x1,y2,x2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1,h1,w2,h2))
                
            # temp_actions=self.anns[sid][src_fid]['actions'][:]
            # bboxes_num.append(len(temp_boxes))
            temp_actions = [Action6to5[i] for i in self.anns[sid][src_fid]['actions'][:]]
            bboxes_num.append(len(temp_boxes))
            
            while len(temp_boxes)!=self.num_boxes:
                temp_boxes.append((0,0,0,0))
                temp_actions.append(-1)
            
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            
            # activities.append(self.anns[sid][src_fid]['group_activity'])
            activities.append(Activity5to4[self.anns[sid][src_fid]['group_activity']])
        
        
        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes=np.array(bboxes,dtype=np.float).reshape(-1,self.num_boxes,4)
        actions=np.array(actions,dtype=np.int32).reshape(-1,self.num_boxes)
        
        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        
        return images, bboxes,  actions, activities, bboxes_num
    
    

    
def return_dataset(cfg):
     
    train_anns=collective_read_dataset(cfg.data_path, cfg.train_seqs)
    train_frames=collective_all_frames(train_anns)

    test_anns=collective_read_dataset(cfg.data_path, cfg.test_seqs)
    test_frames=collective_all_frames(test_anns)

    training_set=CollectiveDataset(train_anns,train_frames,cfg.data_path,cfg.image_size,cfg.out_size,num_frames = 765, is_training=True,is_finetune=(cfg.training_stage==1))
                                          
                                          

    validation_set=CollectiveDataset(test_anns,test_frames,cfg.data_path,cfg.image_size,cfg.out_size,num_frames = 765, is_training=False,is_finetune=(cfg.training_stage==1))
                                          
                                          

           


    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))

    return training_set, validation_set



def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 4
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={'volleyball':Basenet_volleyball, 'collective':Basenet_collective}
    gcnnet_list={'volleyball':GCNnet_volleyball, 'collective':GCNnet_collective}
    
    if cfg.training_stage==1:
        Basenet=basenet_list[cfg.dataset_name]
        model=Basenet(cfg)
    elif cfg.training_stage==2:
        GCNnet=gcnnet_list[cfg.dataset_name]
        model=GCNnet(cfg)
        # Load backbone
        model.loadmodel(cfg.stage1_model_path)
    else:
        assert(False)
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    train_list={'volleyball':train_volleyball, 'collective':train_collective}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        test_info=test(validation_loader, model, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result={'epoch':0, 'activities_acc':0}
    start_epoch=1
    
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)
        
        
        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, model, device, epoch, cfg)
            show_epoch_info("epoch done ",epoch)
            
            
    
   


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        # forward
        actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
        
        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))
        bboxes_num=batch_data[4].reshape(batch_size,num_frames)

        actions_in_nopad=[]
        if cfg.training_stage==1:
            actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])
        else:
            for b in range(batch_size):
                N=bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        if cfg.training_stage==1:
            activities_in=activities_in.reshape(-1,)
        else:
            activities_in=activities_in[:,0].reshape(batch_size,)
        
        # Predict actions
        actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)  
        actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

        # Predict activities
        activities_loss=F.cross_entropy(activities_scores,activities_in)
        activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
        activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        
        
        # Get accuracy
        actions_accuracy=actions_correct.item()/actions_scores.shape[0]
        activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return train_info
        
    
def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    
    """
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data=[b.to(device=device) for b in batch_data]
            print("batch_data",batch_data)
    """
    count_of_frames = 0
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            
            batch_data=[b.to(device=device) for b in batch_data]
            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data[4].reshape(batch_size,num_frames)

            # forward
            actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
            
            """
            print("batch_data[0]",batch_data[0].shape)
            print("bata_data[1]",batch_data[1].shape)
            print("batch_data[4]",batch_data[4].shape)
            print("batch_size",batch_size)
            print("num_frames",num_frames)
            """
            
            actions_in_nopad=[]
            
            if cfg.training_stage==1:
                actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
                bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
                for bt in range(batch_size*num_frames):
                    N=bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt,:N])
            else:
                for b in range(batch_size):
                    N=bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            if cfg.training_stage==1:
                activities_in=activities_in.reshape(-1,)
            else:
                activities_in=activities_in[:,0].reshape(batch_size,)

            actions_loss=F.cross_entropy(actions_scores,actions_in)  
            actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
            actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

            # Predict activities
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1)  #B,
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            ###
            print("activity_labels_size",activities_labels.size())
            print("activity_in_size",activities_in.size())
            count_of_frames+=activities_labels.size()
            ###
            
            # Get accuracy
            actions_accuracy=actions_correct.item()/actions_scores.shape[0]
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)
            
           
                
                
            
            #dict_epoch[epoch].append([actions_in,actions_labels,activities_in,activities_labels])
            
            #dict_epoch[epoch] = [actions_in,actions_labels,activities_in,activities_labels]

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    
   
    
    
    
    """
    print(actions_in,"actio_in")
    print(actions_labels,"actio_lab")
    print(activities_in,"activ_in")
    print(activities_labels,"activ_lab")
    
    """
    print("count_of_frames",count_of_frames)
    
    return test_info



cfg=Config('collective')

cfg.device_list="0,1"
cfg.training_stage=1
cfg.train_backbone=True

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=6
cfg.num_activities=5
cfg.num_frames=2

cfg.batch_size=1
cfg.test_batch_size=1
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.exp_note='Collective_stage1'
train_net(cfg)

