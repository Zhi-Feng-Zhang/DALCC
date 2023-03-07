#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import torchvision
import argparse
import os
from dataset.fold_mdlcc import *
import model.paper_alexnet as fc4_model
import evaluation.metrics as metrics
import  random

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
pid = os.getpid()
print(pid)
seed_torch(42)



Start=None
Work_place=None
ALPHA_set=0.03
print("ALPHA===",ALPHA_set)

Target_camera=11
  # 1:CC_1   2:CC-2   3:NUS:1   4:NUS:2   5:NUS:3   6:NUS:4   7:NUS:5  
  #  8:NUS:6   9:NUS:7   10:NUS:8   11:Cube

def parse_args():
    parser = argparse.ArgumentParser(description="Train the achromatic pixel detector")
    a = parser.add_argument
    a("--output-dir", default=Work_place)
    a("--epochs", type=int, default=3000, help="Number of training epochs")
    a("--start-from", default=None)
    a("--batch-size", type=int, default=16, help="Size of the minibatch")
    a("--learning-rate", type=float, default=0.0001, help="Learning rate")
    a("--validate_every", type=int, default=1)
    a("--weight-decay", type=float, default=1e-7, help="Weight decay")
    a("--num-workers", type=int, default=16, help="Number of parallel threads")
    a("--device", default="cuda:0", help="Processing device")
    return parser.parse_args()
def initialization(path, model, optimizer):
    try:
        data = torch.load(path)
    except FileNotFoundError:
        print("Starting from epoch 1")
        return 0
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    print("Continue from epoch", data["epoch"])
    return model,optimizer,data["epoch"]

def save_model(path, model, optimizer, epoch, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
    }, path + ".temp")
    os.rename(path + ".temp", path)  



def main():
    args = parse_args()
    print(args)
    net = fc4_model.CNNModel()
    print(net)
    net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    alpha = ALPHA_set
    epoch_S=0
    if Start is not None:
        net,optimizer,epoch_S=initialization(Start,net,optimizer)

    print("begin trainng")


    for epoch in range(epoch_S,args.epochs):
        target_number=Target_camera   #Cube
        list_camare=[1,2,3,4,5,6,7,8,9,10,11]  #CC_1+NUS_8

        for camere in range(1,12): #CC1+NUS8+Cube1
            if camere not  in list_camare or camere==target_number:
                continue
            dataloader_source=data_loader_train(camere)
            dataloader_target = data_loader_train(target_number)
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)
            i = 0
            angle_loss_history = []
            while i < len_dataloader:
                imgs_S, GTs_S,device_index_S=data_source_iter.next()
                imgs_T, GTs_T, device_index_T = data_target_iter.next()
                imgs=torch.cat([imgs_S, imgs_T], dim=0)
                GTs=torch.cat([GTs_S, GTs_T], dim=0)
                device_index=torch.cat([device_index_S, device_index_T], dim=0)
                domain_label = torch.zeros([imgs_S.shape[0] + imgs_T.shape[0]]).cuda()
                domain_label[:imgs_S.shape[0]] = 1


                imgs = imgs.to(args.device).type(torch.cuda.FloatTensor)
                GTs = GTs.to(args.device).type(torch.cuda.FloatTensor)
                device_index=device_index.to(args.device).type(torch.cuda.FloatTensor)
                domain_label = domain_label.to(args.device).type(torch.cuda.FloatTensor)



                optimizer.zero_grad()
                domain_output, pred, pred_common = net(imgs, device_index, alpha,k=camere-1)
                loss_angle=metrics.angle_loss(pred,GTs)
                loss_cross=metrics.Cross_Loss(domain_output,domain_label)
                loss=loss_angle+loss_cross
                loss.backward()
                optimizer.step()

                angle_loss_history.append(loss_angle.item())
                i += 1

            angle_loss = sum(angle_loss_history) / max(1, len(angle_loss_history))
            print("epoch: ",epoch,"angle loss: ",angle_loss,"*********camrea:",camere)


        if epoch % args.validate_every == 0:
            dataloader_test = data_loader_test(Target_camera)
            net.eval()
            val_errors = []
            with torch.no_grad():
                for images, illums, device_index in dataloader_test:
                    images = images.to(args.device).type(torch.cuda.FloatTensor)
                    illums = illums.to(args.device).type(torch.cuda.FloatTensor)
                    device_index = device_index.to(args.device).type(torch.cuda.FloatTensor)
                    domain_output, preds,pred_common = net(images,device_index, 0,target_number-1)
                    loss = metrics.angle_loss(preds, illums)
                    val_errors.append(loss.item())

                angle_loss = sum(val_errors) / max(1, len(val_errors))
                print("epoch: ", epoch,"Camera: ",camere, metrics.metric(angle_loss))
                
                    
if __name__ == "__main__":
    main()




