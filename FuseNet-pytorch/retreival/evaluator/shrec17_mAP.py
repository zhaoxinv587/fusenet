'''
1. 抽取测试集object特征
2. 每个object与其他物体计算相似度分数，形成排序列表
3. 用shrec17提供的代码评测检索效果
'''

import os, sys
import numpy as np
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch
from FuseNet.models.pointnet_Shrec17 import get_model
from FuseNet.models.FuseNet_Shrec17 import FuseNet, FC, W, W1
from FuseNet.shapenetcore55dataset import ShapeNetCore55_MultiView, ShapeNetCore55_Point, MyDataset
def get_precision_and_recall(y_label, y_prob,y_pred_label):
    """

    :param y_label: true label
    :param y_score: predict probability
    :param n:
    :return:
    """

    assert len(y_label) == len(y_prob)
    # invert sort y_pred
    # score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    # y_prob = np.array(y_prob)[score_indices]
    # y_true = np.array(y_label)[score_indices]
    # y_pred_label = np.array(y_pred_label)[score_indices]
    # print(y_label)
    # print(y_pred_label)
    for i in range(len(y_pred_label)):
        if y_pred_label[i] == y_label[0]:
            y_pred_label[i] = 1
        else:
            y_pred_label[i] = 0

    # print(y_pred_label)
    # for i in range(len(y_pred_label)):
    #     y_pred_label[i] = 0
    # for i in range(100):
    #     y_pred_label[i] = 1
    # ------------------get tps and fps at distinct value -------------------------
    # extract the indices associated with the distinct values
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_pred_label.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_pred_label)[threshold_idxs]
    # computer false positive
    fps = threshold_idxs + 1 - tps
    #------------------------ computer precision and recall------------------------
    # computer precision
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    # computer recall
    recall = tps / tps[-1]

    # stop when full recall attained
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind+1)

    # add (0, 1) vertex to P-R curve
    final_precision = np.r_[1, precision[sl]]
    final_recall = np.r_[0, recall[sl]]
    #------------------------ computer AP------------------------------------------
    height = np.diff(final_recall)
    bottom = np.convolve(final_precision, v=[1, 1], mode='valid')
    ap = np.sum(height * bottom / 2)

    return final_precision, final_recall, ap


if __name__ == '__main__':



    '''Model loading'''
    # 点云
    modelpoint = get_model(55, normal_channel=False).cuda()
    modelpoint.load_state_dict(torch.load('/home/ma2/zx/FuseNet/log/Points/shrec17_pointnet.pth'))
    # 多视图
    modelmutiview = FuseNet().cuda()
    modelmutiview.load_state_dict(torch.load('/home/ma2/zx/FuseNet/log/Mutiview/Shrec17_vgg16.pth'))
    # FC
    modelFC = FC().cuda()
    modelFC.load_state_dict(torch.load('/home/ma2/zx/FuseNet/log/FC/FC_shrec17.pth'))
    # W
    modelW = W().cuda()
    modelW.load_state_dict(torch.load('/home/ma2/zx/FuseNet/log/W/W_shrec17.pth'))
    # W1
    modelW1 = W1().cuda()
    modelW1.load_state_dict(torch.load('/home/ma2/zx/FuseNet/log/W1/W1_shrec17.pth'))

    '''Data loading'''
    # 多视图

    test_Mutiviewdataset = ShapeNetCore55_MultiView(root_dir='/home/ma2/zx/FuseNet/data/shrec17MutiView', label_file='test.csv',
                                                    version='normal', num_views=12, total_num_views=12, num_classes=55)

    # 点云

    test_Pointsdataset = ShapeNetCore55_Point(root_dir='/media/ma2/3EEF1FF3BAE5C446/shrec17Points',
                                              label_file='test.csv', version='normal', num_classes=55)

    testDataset = MyDataset(dataset1=test_Mutiviewdataset, dataset2=test_Pointsdataset)
    testdataloader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=16, shuffle=False, pin_memory=True)
    '''Eval'''

    print('num_test_files(Multiview): ', len(test_Mutiviewdataset))
    print('num_test_files(Points): ', len(test_Pointsdataset))

    modelmutiview.eval()
    modelpoint.eval()  # ，model.eval()是保证BN用全部训练数据的均值和方差
    modelFC.eval()
    modelW.eval()
    modelW1.eval()

    correct, total = 0, 0
    all_true = []
    all_prob = []
    all_pred_label = []

    for _, data in enumerate(testdataloader):
        datamutiview = data[0]
        datapoint = data[1]
        labels, inputs, shapname1 = datamutiview
        target, points, shapname2 = datapoint
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        with torch.no_grad():
            p1, p2, p3 = modelpoint(points.float())
        labels = labels.cuda()
        inputs = torch.squeeze(inputs, 0).cuda()
        in_data = inputs.cuda().view(-1, 1, 224, 224)
        with torch.no_grad():
            m1a, m1m, m3a, m3m, m5a, m5m = modelmutiview(in_data)  # 8 25088

        m1, m3, m5 = modelW(m1a, m1m, m3a, m3m, m5a, m5m)
        feature1 = torch.cat((p1, m1), 1)
        feature2 = torch.cat((p2, m3), 1)
        feature3 = torch.cat((p3, m5), 1)
        fuse = modelW1(feature1, feature2, feature3)
        fuse = modelFC(fuse)
        pred = fuse.argmax(dim=1)
        # print(pred)
        # print(labels)
        # print(target)
        pred_label = fuse.argmax(dim=1).cpu().numpy().tolist()
        # gg.append(test)
        true_label = labels.cpu().numpy().tolist()
        pred_probablity = torch.nn.functional.softmax(fuse, dim=1).max(dim=1)[0].cpu().detach().numpy().tolist()
        # print(pred_probablity)
        all_true += true_label
        all_prob += pred_probablity
        all_pred_label += pred_label
        correct += torch.eq(pred, labels).sum().item()
        total += len(labels)

    print('Accuracy: {}'.format(correct / total))
    # nextVal = sorted(gg, reverse=True)

    all_true = np.array(all_true)
    all_prob = np.array(all_prob)
    all_pred_label = np.array(all_pred_label)



    precision, recall, ap = get_precision_and_recall(all_true, all_prob,all_pred_label)
    print(precision)
    print(recall)
    print(ap)









