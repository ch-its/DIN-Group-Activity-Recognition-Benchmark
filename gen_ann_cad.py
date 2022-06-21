import sys
sys.path.append(".")

from copy import deepcopy
import torch
import torch.optim as optim

import time
import random
import os
import sys

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from base_model import *
from utils import *

from annotate_cad import *

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
        'num_workers': 4, # 4,
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)


    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={ 'collective':Basenet_collective}
    gcnnet_list={'dynamic_collective':Dynamic_collective}
                 
    
   
    GCNnet = gcnnet_list[cfg.inference_module_name]
    model = GCNnet(cfg)
    #model.loadmodel("result/stage1_epoch1_76.58%.pth")
    state = torch.load("stage2_epoch9_94.59%.pth")
    
    mydict = deepcopy(state['state_dict'])
    for i in state['state_dict']:
        k_new = '.'.join(i.split('.')[1:])
        mydict[k_new] = mydict.pop(i)
    
    del state
    model.load_state_dict(mydict,strict=True)
    #model.loadmodel("result/[Dynamic Volleyball_stage2_res18_litedim128_reproduce_1_stage2]<2022-05-26_21-02-11>/stage2_epoch6_91.89%.pth")
    #model.load_state_dict(torch.load("result/[Dynamic Volleyball_stage2_res18_litedim128_reproduce_1_stage2]<2022-05-26_21-02-11>/stage2_epoch6_91.89%.pth"),strict=True)
    
    #model.loadmodel("result/[Dynamic Volleyball_stage2_res18_litedim128_reproduce_1_stage2]<2022-05-26_21-02-11>/stage2_epoch6_91.89%.pth")
    
   
    
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.eval()
    

    
    
    
    

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0}
    start_epoch = 1
    acc_cnt = 0
    total_cnt = 0
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        
        acc_cnt = 0
        
        
        
        
            
        # One epoch of forward and backward
        
        start = time.time()
       
        with torch.no_grad():
            
            for batch_data in validation_loader:
                
                # prepare batch data
                batch_data=[b.to(device=device) for b in batch_data]
                batch_size=batch_data[0].shape[0]
                num_frames=batch_data[0].shape[1]

                actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
                activities_in=batch_data[3].reshape((batch_size,num_frames))
                bboxes_num=batch_data[4].reshape(batch_size,num_frames)

                # forward
                activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))
                
                if cfg.training_stage==1:
                    activities_in=activities_in.reshape(-1,)
                else:
                    activities_in=activities_in[:,0].reshape(batch_size,)

            # actions_loss=F.cross_entropy(actions_scores,actions_in)
            # actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
            # actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
            # actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            # actions_meter.update(actions_accuracy, actions_scores.shape[0])

            # Predict activities
                
                activities_labels=torch.argmax(activities_scores,dim=1)  #B,
                
                #print(ret['activities'])
                # Predict actions
                
                
                activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                acc_cnt+=activities_correct
                total_cnt += activities_scores.shape[0]
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                
                
                
                if not os.path.isdir('ann_cad_'+str(cfg.test_seqs[0])):
                        os.makedirs('ann_cad_'+str(cfg.test_seqs[0]))

                for i in cfg.test_seqs:


                    for j in range(len(activities_in)):


                        f= open('ann_cad_'+str(i)+"/"+"anns__"+str(cfg.test_seqs[0])+"epoch__"+str(epoch)+".txt","a+")
                        f.write(str(activities_in[j].item())+","+str(activities_labels[j].item()))
                        f.write("\n")
                        f.close()
                end = time.time()
                
                
        print("time elapsed is",end-start)
        print("accuracy is : ",(acc_cnt/total_cnt)*100)      
        
    return 'ann_cad_'+str(cfg.test_seqs[0])+"/"+"anns__"+str(cfg.test_seqs[0])+"epoch__"+str(1)+".txt"
    

            
           
           
           
   


    






def pipeline(seqno):
    
    cfg=Config('collective')
    cfg.inference_module_name = 'dynamic_collective'


    cfg.device_list="0,1,2"
    cfg.training_stage=2
    cfg.use_gpu = True
    cfg.use_multi_gpu = True
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True

    # ResNet18
    cfg.backbone = 'vgg16'
    cfg.image_size = 480, 720
    cfg.out_size = 15, 23
    cfg.emb_features = 512
    #cfg.stage1_model_path = ''
    

    # VGG16
    # cfg.backbone = 'vgg16'
    # cfg.image_size = 480, 720
    # cfg.out_size = 15, 22
    # cfg.emb_features = 512
    # cfg.stage1_model_path = 'result/basemodel_CAD_vgg16.pth'

    cfg.num_boxes = 13
    cfg.num_actions = 5
    cfg.num_activities = 4
    cfg.num_frames = 10
    cfg.num_graph = 4
    cfg.tau_sqrt=True
    cfg.batch_size = 2
    cfg.test_batch_size = 8
    cfg.test_interval_epoch = 1
    cfg.train_learning_rate = 5e-5
    cfg.train_dropout_prob = 0.5
    cfg.weight_decay = 1e-4
    cfg.lr_plan = {}
    cfg.max_epoch = 100


    # Dynamic Inference setup
    cfg.group = 1
    cfg.stride = 1
    cfg.ST_kernel_size = (3, 3)
    cfg.dynamic_sampling = True
    cfg.sampling_ratio = [1]  # [1,2,4]
    cfg.lite_dim = None # 128
    cfg.scale_factor = True
    cfg.beta_factor = False
    cfg.hierarchical_inference = False
    cfg.parallel_inference = False
    
    cfg.test_seqs=[3]
    
    cfg.exp_note='Dynamic_collective'
    train_net(cfg)

    fp = train_net(cfg)
    
    ann(seqno,fp)



#seq = input("Enter the sequence number for which the video is to be generated")
pipeline(3)