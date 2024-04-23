import  torch
from torch import nn, optim
from FuseNet import provider
from FuseNet.models.pointnet_Shrec17 import get_model
from FuseNet.models.FuseNet_Shrec17 import FuseNet ,FC,W,W1
from shapenetcore55dataset import ShapeNetCore55_MultiView, ShapeNetCore55_Point, MyDataset

if __name__ == '__main__':

    '''Model loading'''
    #点云
    modelpoint = get_model(55, normal_channel=False).cuda()

    #多视图
    modelmutiview = FuseNet().cuda()

    #FC
    modelFC = FC().cuda()

    #W
    modelW = W().cuda()

    #W1
    modelW1 = W1().cuda()


    '''Data loading'''
    #多视图
    train_Mutiviewdataset = ShapeNetCore55_MultiView(root_dir='data/shrec17MutiView', label_file='train.csv',
                 version='normal', num_views=12, total_num_views=12, num_classes=55)


    test_Mutiviewdataset = ShapeNetCore55_MultiView(root_dir='data/shrec17MutiView', label_file='test.csv',
                 version='normal', num_views=12, total_num_views=12, num_classes=55)

    #点云
    train_Pointsdataset= ShapeNetCore55_Point(root_dir=', label_file='train.csv', version='normal', num_classes=55)

    test_Pointsdataset= ShapeNetCore55_Point(root_dir='/shrec17Points', label_file='test.csv', version='normal', num_classes=55)

    trainDataset = MyDataset(dataset1=train_Mutiviewdataset, dataset2=train_Pointsdataset)
    traindataloader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=8, shuffle=True, pin_memory=True)

    testDataset = MyDataset(dataset1=test_Mutiviewdataset, dataset2=test_Pointsdataset)
    testdataloader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=8, shuffle=True, pin_memory=True)


    '''Train'''
    # print('num_train_files(Multiview): ' , len(train_Mutiviewdataset))
    # print('num_train_files(Points): ', len(train_Pointsdataset))
    #
    # modelpoint.eval()
    # modelmutiview.eval()
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
    # EPOCHES = 8
    # modelW.train()
    # modelW1.train()
    # modelFC.train()
    # for ep in range(EPOCHES):
    #     batch_id = 1
    #     correct, total, total_loss = 0, 0, 0.
    #     for _,data in enumerate(traindataloader):
    #         datamutiview = data[0]
    #         datapoint = data[1]
    #         labels, inputs,shapname1 = datamutiview
    #         target ,points,shapname2= datapoint
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
    #         p1,p2,p3 = modelpoint(points)
    #
    #         #多视图
    #         labels = labels.cuda()
    #         in_data = inputs.cuda().view(-1,1,224,224)
    #         m1a ,m1m , m3a,m3m, m5a , m5m  = modelmutiview(in_data) #8 25088
    #         m1,m3,m5 = modelW(m1a,m1m,m3a,m3m,m5a,m5m)
    #
    #         #特征融合
    #         feature1 = torch.cat((p1,m1),1)
    #         feature2 = torch.cat((p2,m3),1)
    #         feature3 = torch.cat((p3,m5),1)
    #         fuse = modelW1(feature1,feature2,feature3)
    #         fuse = modelFC(fuse)
    #
    #         # Compute loss & accuracy
    #         loss = criterion(fuse, labels).cuda()
    #         pred = fuse.argmax(dim=1)  # 返回每一行中最大值元素索引
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
    #     torch.save(modelFC.state_dict(), 'log/FC/FC_shrec17.pth')
    #     torch.save(modelW.state_dict(), 'log/W/W_shrec17.pth')
    #     torch.save(modelW1.state_dict(), 'log/W1/W1_shrec17.pth')


    '''Eval'''

    print('num_test_files(Multiview): ' ,len(test_Mutiviewdataset))
    print('num_test_files(Points): ' ,len(test_Pointsdataset))

    modelmutiview.eval()
    modelpoint.eval()  
    modelFC.eval()
    modelW.eval()
    modelW1.eval()

    correct, total = 0, 0
    for _, data in enumerate(testdataloader):
        datamutiview = data[0]
        datapoint = data[1]
        labels, inputs, shapname1 = datamutiview
        target, points, shapname2 = datapoint
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        with torch.no_grad():
            p1,p2,p3 = modelpoint(points.float())
        labels = labels.cuda()
        inputs = torch.squeeze(inputs , 0).cuda()
        in_data = inputs.cuda().view(-1, 1, 224, 224)
        with torch.no_grad():
            m1a ,m1m , m3a,m3m, m5a , m5m  = modelmutiview(in_data) #8 25088

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
