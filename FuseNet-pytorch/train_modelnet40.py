from sklearn.metrics import classification_report
from torch import nn, optim

from FuseNet import provider
from FuseNet.models.FuseNet_ModelNet40 import FuseNet ,FC,W,W1
from modelnet40dataset import MultiviewImgDataset, ModelNetDataLoader, MyDataset
import argparse
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='PointNet_modelnet10',help='Experiment root') #pointnet2_cls_msg
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate Points scores with voting')
    return parser.parse_args()


def main(args):

    def log_string(str):
        logger.info(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = 'log/Points/' + args.log_dir

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    # 点云
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_images_points/modelnet40_normal_resampled/'
    train_Pointsdataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=False)
    val_Pointsdataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)

    #多视图
    testpath = "data/modelnet40_images_12views/*/test"
    trainpath = "data/modelnet40_images_12views/*/train"
    train_Mutiviewdataset = MultiviewImgDataset(trainpath)
    val_Mutiviewdataset = MultiviewImgDataset(testpath)

    trainDataset = MyDataset(dataset1=train_Mutiviewdataset, dataset2=train_Pointsdataset)
    traindataloader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=8, shuffle=True,drop_last=True)

    testDataset = MyDataset(dataset1=val_Mutiviewdataset, dataset2=val_Pointsdataset)
    testdataloader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=8, shuffle=False, pin_memory=True,drop_last=True)
    '''MODEL LOADING'''
    #点云
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module('pointnet_ModelNet')
    model1= model.get_model(num_class, normal_channel=args.use_normals)
    model1 = model1.to(device)

    #多视图
    model2 = FuseNet().to(device)


    #FC
    modelFC = FC().to(device)


    #权重W
    modelW = W().to(device)


    #权重W1
    modelW1 = W1().to(device)



    # '''Train'''
    # print('num_train_files(Multiview): ' + str((len(train_Mutiviewdataset.filepaths))/12))
    # model1.eval()
    # model2.eval()  
    #
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # optFC = optim.SGD(modelFC.parameters(), lr=0.01, momentum=0.8, weight_decay=0.001)
    # schedulerFC = optim.lr_scheduler.ExponentialLR(optFC, gamma=0.96)
    # optW = optim.Adam(modelW.parameters(), lr=0.001)
    # schedulerW = optim.lr_scheduler.ExponentialLR(optW, gamma=0.96)
    # optW1 = optim.Adam(modelW1.parameters(), lr=0.001)
    # schedulerW1 = optim.lr_scheduler.ExponentialLR(optW1, gamma=0.96)
    #
    #
    #
    # EPOCHES = 20
    # modelFC.train()
    # modelW.train()
    # modelW1.train()
    # #modelAFF.train()
    # #modelMS_CAM.train()
    # for ep in range(EPOCHES):
    #     batch_id = 1
    #     correct, total, total_loss = 0, 0, 0.
    #     for _,data in enumerate(traindataloader):
    #         datamutiview = data[0]
    #         datapoint = data[1]
    #         labels, inputs = datamutiview
    #         points, target = datapoint
    #         optW.zero_grad()
    #         optFC.zero_grad()
    #         optW1.zero_grad()
    #         #点云
    #         points = points.data.numpy()
    #         points = provider.random_point_dropout(points)
    #         points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    #         points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    #         points = torch.Tensor(points)
    #         points = points.transpose(2, 1)
    #         points, target = points.cuda(), target.cuda()
    #         p1,p2,p3 = model1(points)
    #         #多视图
    #         labels = labels.to(device)
    #         in_data = inputs.to(device).view(-1,1,224,224)
    #         m1a ,m1m , m3a,m3m, m5a , m5m  = model2(in_data) #8 25088
    #         m1,m3,m5 = modelW(m1a,m1m,m3a,m3m,m5a,m5m)
    #
    #         #特征融合
    #         feature1 = torch.cat((p1,m1),1)
    #         feature2 = torch.cat((p2,m3),1)
    #         feature3 = torch.cat((p3,m5),1)
    #         fuse = modelW1(feature1,feature2,feature3)
    #
    #         fuse = modelFC(fuse)
    #
    #
    #         # Compute loss & accuracy
    #         loss = criterion(fuse, labels).to(device)
    #         pred = fuse.argmax(dim=1)  # 返回每一行中最大值元素索引
    #
    #         correct += torch.eq(pred, labels).sum().item()
    #         total += len(labels)
    #         accuracy = correct / total
    #         total_loss += loss
    #         loss.backward()  # compute gradient (backpropagation)
    #         optW.step()
    #         optFC.step() # update model with optimizer
    #         optW1.step()
    #         print('Epoch {}, batch {}, train_loss: {}, train_accuracy: {}'.format(ep + 1,
    #                                                                               batch_id,
    #                                                                               total_loss / batch_id,
    #                                                                               accuracy))
    #         batch_id += 1
    #     schedulerFC.step(ep)
    #     schedulerW.step(ep)
    #     schedulerW1.step(ep)
    #     print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
    #     # Save model
    #     torch.save(modelFC.state_dict(), 'log/FC/FC_modelnet10.pth')
    #     torch.save(modelW.state_dict(), 'log/W/W_modelnet10.pth')
    #     torch.save(modelW1.state_dict(), 'log/W1/W1_modelnet10.pth')
    #torch.save(modelFC1.state_dict(), 'log/FC1/FC_2048_early.pth')



    '''Eval'''

    print('num_test_files(Multiview): ' + str((len(val_Mutiviewdataset.filepaths))/12))

    model1.eval()
    model2.eval()  
    modelFC.eval()
    modelW.eval()
    modelW1.eval()

    correct, total = 0, 0
    for _, data in enumerate(testdataloader):
        datamutiview = data[0]
        datapoint = data[1]
        labels, inputs = datamutiview
        points, target = datapoint
        points, target = points.cuda(), target.cuda()
        points = points.transpose(2, 1)
        with torch.no_grad():
            p1,p2,p3 = model1(points)
        labels = labels.to(device)
        inputs = torch.squeeze(inputs , 0).to(device)
        in_data = inputs.to(device).view(-1, 1, 224, 224)
        with torch.no_grad():
            m1a ,m1m , m3a,m3m, m5a , m5m  = model2(in_data) #8 25088

        m1,m3,m5 = modelW(m1a,m1m,m3a,m3m,m5a,m5m)
        feature1 = torch.cat((p1, m1), 1)
        feature2 = torch.cat((p2, m3), 1)
        feature3 = torch.cat((p3, m5), 1)
        fuse = modelW1(feature1, feature2, feature3)

        fuse = modelFC(fuse)
        pred = fuse.argmax(dim=1)

        correct += torch.eq(pred, labels).sum().item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))






if __name__ == '__main__':
    args = parse_args()
    main(args)
