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

from annotate import *

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
    basenet_list={ 'volleyball':Basenet_volleyball}
    gcnnet_list={'dynamic_volleyball':Dynamic_volleyball}
                 
    
   
    GCNnet = gcnnet_list[cfg.inference_module_name]
    model = GCNnet(cfg)
    #model.loadmodel("result/stage1_epoch1_76.58%.pth")
    state = torch.load("result/[Dynamic Volleyball_stage2_res18_litedim128_reproduce_1_stage2]<2022-05-26_21-02-11>/stage2_epoch6_91.89%.pth")
    
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
            
            for batch_data_test in validation_loader:
                
                # prepare batch data
                batch_data_test=[b.to(device=device) for b in batch_data_test]
                batch_size=batch_data_test[0].shape[0]
                num_frames=batch_data_test[0].shape[1]

                actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
                activities_in=batch_data_test[3].reshape((batch_size,num_frames))

                # forward
                # actions_scores,activities_scores=model((batch_data_test[0],batch_data_test[1]))
                # activities_scores = model((batch_data_test[0], batch_data_test[1]))
                
                ret = model((batch_data_test[0], batch_data_test[1]))
                
                #print(ret['activities'])
                # Predict actions
                actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
                activities_in=activities_in[:,0].reshape((batch_size,))
                
                activities_scores = ret['activities']
                #print(activities_scores)
                activities_labels = torch.argmax(activities_scores,dim=1)
                
                activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                acc_cnt+=activities_correct
                total_cnt += activities_scores.shape[0]
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                
                
                
                if not os.path.isdir('ann_vb_'+str(cfg.test_seqs[0])):
                        os.makedirs('ann_vb_'+str(cfg.test_seqs[0]))

                for i in cfg.test_seqs:


                    for j in range(len(activities_in)):


                        f= open('ann_vb_'+str(i)+"/"+"anns__"+str(cfg.test_seqs[0])+"epoch__"+str(epoch)+".txt","a+")
                        f.write(str(activities_in[j].item())+","+str(activities_labels[j].item()))
                        f.write("\n")
                        f.close()
                end = time.time()
                
                
        print("time elapsed is",start-end)
        print("accuracy is : ",(acc_cnt/total_cnt)*100)      
        
    return 'ann_vb_'+str(cfg.test_seqs[0])+"/"+"anns__"+str(cfg.test_seqs[0])+"epoch__"+str(1)+".txt"
    

            
           
           
           
   


    






def pipeline(seqno):
    
    


    cfg=Config('volleyball')
    cfg.inference_module_name = 'dynamic_volleyball'

    cfg.device_list = "0,1,2"
    cfg.use_gpu = True
    cfg.use_multi_gpu = True
    cfg.training_stage = 2
    cfg.train_backbone = True
    cfg.test_before_train = False
    cfg.test_interval_epoch = 1
    cfg.test_seqs = [seqno]  #video id list of test set
    cfg.train_seqs = [ s for s in range(1,52) if s not in cfg.test_seqs]  #video id list of train set 
    # vgg16 setup
    cfg.backbone = 'vgg16'
    cfg.stage1_model_path = 'result/[Volleyball_stage1_stage1]<2022-05-11_06-12-49>/stage1_epoch1_76.58%.pth'
    cfg.out_size = 22, 40
    cfg.emb_features = 512

    # res18 setup
    # cfg.backbone = 'res18'
    # cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
    # cfg.out_size = 23, 40
    # cfg.emb_features = 512

    # Dynamic Inference setup
    cfg.group = 1
    cfg.stride = 1
    cfg.ST_kernel_size = [(3, 3)] #[(3, 3),(3, 3),(3, 3),(3, 3)]
    cfg.dynamic_sampling = True
    cfg.sampling_ratio = [1]
    cfg.lite_dim = 128 # None # 128
    cfg.scale_factor = True
    cfg.beta_factor = False
    cfg.hierarchical_inference = False
    cfg.parallel_inference = False
    cfg.num_DIM = 1
    cfg.train_dropout_prob = 0.3

    cfg.batch_size = 2
    cfg.test_batch_size = 1
    cfg.num_frames = 10
    cfg.load_backbone_stage2 = True
    cfg.train_learning_rate = 1e-4
    # cfg.lr_plan = {11: 3e-5, 21: 1e-5}
    # cfg.max_epoch = 60
    # cfg.lr_plan = {11: 3e-5, 21: 1e-5}
    cfg.lr_plan = {11: 1e-5}
    cfg.max_epoch = 1
    cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

    cfg.exp_note = 'Dynamic Volleyball_stage2_res18_litedim128_reproduce_1'
    fp = train_net(cfg)
    
    ann(seqno,fp)



#seq = input("Enter the sequence number for which the video is to be generated")
pipeline(4)