import  torch
from torch import nn, optim

from pointnet_Shrec17 import get_model
from FuseNet.models.FuseNet_Shrec17 import FuseNet
from FuseNet.shapenetcore55dataset import ShapeNetCore55_MultiView, ShapeNetCore55_Point, MyDataset

if __name__ == '__main__':


    '''Data loading'''
    #多视图
    train_Mutiviewdataset = ShapeNetCore55_MultiView(root_dir='../data/shrec17MutiView', label_file='train.csv',
                                                     version='normal', num_views=12, total_num_views=12, num_classes=55)
    train_Mutiviewloader = torch.utils.data.DataLoader(train_Mutiviewdataset, batch_size=1, shuffle=True, num_workers=0)


    #点云
    train_Pointsdataset= ShapeNetCore55_Point(root_dir='../data/shrec17Points', label_file='train.csv', version='normal', num_classes=55)
    train_PointsLoader = torch.utils.data.DataLoader(train_Pointsdataset, batch_size=1, shuffle=True)

    myDataset = MyDataset(dataset1=train_Mutiviewdataset, dataset2=train_Pointsdataset)
    dataloader = torch.utils.data.DataLoader(dataset=myDataset, batch_size=1, shuffle=False, pin_memory=True)


    '''Train'''
    print('num_train_files(Multiview): ' , len(train_Mutiviewdataset))
    print('num_train_files(Points): ', len(train_Pointsdataset))



    for _, data in enumerate(dataloader):

        data1, data2 = data[0], data[1]
        labels ,data ,shapname1= data1
        target , inputs , shapname2 = data2
        if shapname1 != shapname2:
            print(shapname1)
            print(shapname2)
            break




