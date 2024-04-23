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
import  torch
from tqdm import tqdm

from FuseNet.models.pointnet_Shrec17 import get_model
from FuseNet.models.FuseNet_Shrec17 import FuseNet ,FC,W,W1
from FuseNet.shapenetcore55dataset import ShapeNetCore55_MultiView, ShapeNetCore55_Point, MyDataset

if __name__ == '__main__':

    '''Model loading'''
    #点云
    modelpoint = get_model(55, normal_channel=False).cuda()
    modelpoint.load_state_dict(torch.load('../log/Points/shrec17_pointnet.pth'))
    #多视图
    modelmutiview = FuseNet().cuda()
    modelmutiview.load_state_dict(torch.load('../log/Mutiview/Shrec17_vgg16.pth'))
    #FC
    modelFC = FC().cuda()
    modelFC.load_state_dict(torch.load('../log/FC/FC_shrec17.pth'))
    #W
    modelW = W().cuda()
    modelW.load_state_dict(torch.load('../log/W/W_shrec17.pth'))
    #W1
    modelW1 = W1().cuda()
    modelW1.load_state_dict(torch.load('../log/W1/W1_shrec17.pth'))

    '''Data loading'''
    #多视图

    test_Mutiviewdataset = ShapeNetCore55_MultiView(root_dir='/media/ma2/3EEF1FF3BAE5C446/data/shrec17_points/shrec17_images_20views', label_file='test.csv',
                                                    version='normal', num_views=12, total_num_views=12, num_classes=55)

    #点云

    test_Pointsdataset= ShapeNetCore55_Point(root_dir='/media/ma2/3EEF1FF3BAE5C446/data/shrec17_points', label_file='test.csv', version='normal', num_classes=55)


    testDataset = MyDataset(dataset1=test_Mutiviewdataset, dataset2=test_Pointsdataset)
    testdataloader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=8, shuffle=False, pin_memory=True)
    '''Eval'''

    print('num_test_files(Multiview): ' ,len(test_Mutiviewdataset))
    print('num_test_files(Points): ' ,len(test_Pointsdataset))

    modelmutiview.eval()
    modelpoint.eval()  # ，model.eval()是保证BN用全部训练数据的均值和方差
    modelFC.eval()
    modelW.eval()
    modelW1.eval()

    correct, total = 0, 0
    pred_logits = []
    pred_class_ids = []
    all_true = []
    all_pred = []

    for _, data in tqdm(enumerate(testdataloader), total=len(testdataloader)):
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
        with torch.no_grad():
            fuse = modelFC(fuse).cpu().numpy()

        logits = softmax(fuse, axis=1)
        class_ids = logits.argmax(axis=1)

        pred_logits.extend(logits)
        pred_class_ids.extend(class_ids)


                # forward pass to get shape `category distribution`
pred_logits = np.array(pred_logits)
pred_class_ids = np.array(pred_class_ids)
print('pred_logits.shape:', pred_logits.shape)
print('pred_class_ids.shape:', pred_class_ids.shape)

                        # --- step 2: compute rank list for each object
                        # 相似度计算方式：
                        #   1. 物体类别分布做排序依据，把同类物体返回，再按照概率大小排序
                        #        理解了 RotationNet 的排序过程，仿照它实现自己的
                        #   2. 物体特征表示计算相似度，形成长为N的rank list

from os.path import basename, join, exists

shape_names = []
filepaths = np.array(test_Mutiviewdataset.filepaths)[::12]  # take one view every `num_views`
print('len(filepaths):', len(filepaths))
label_file = 'test.csv'
mode = os.path.splitext(label_file)[0]
                # NOTE 生成 test.csv 和 fp_test.txt
with open(label_file, 'w') as fout1, open(f'fp_{mode}.txt', 'w') as fout2:
        fout1.write('id,synsetId,subSynsetId,modelId,split\n')
        for fp in filepaths:
            filename = basename(fp)
            shape_name = filename[:6]
            shape_names.append(shape_name)

            class_name = test_Mutiviewdataset.shape2class[shape_name]
            class_id = test_Mutiviewdataset.classnames.index(class_name)
            fout1.write(f'{shape_name},{class_id},{class_id},{shape_name},{mode}\n')
            fout2.write(f'{fp}\n')

shape_names = np.array(shape_names)
num_objects = len(pred_logits)
method_name = '0'
savedir = join('evaluator', f'{mode}_normal')
if not exists(savedir):
        os.mkdir(savedir)

for idx in range(num_objects):
    filename = join(savedir, shape_names[idx])
    with open(filename, 'w') as fout:  # viewformer/test_normal/000009
        scores_column = pred_logits[:, pred_class_ids[idx]].copy()
        scores_column[idx] = float('inf')

                                # pick up sample ids in the same class
        ids = [i for i, class_id in enumerate(pred_class_ids) if class_id == pred_class_ids[idx]]
        scores_column_ = scores_column[ids]  # scores_column_ 的长度不一定 >=1000，往往是 <1000
        shape_names_ = shape_names[ids]
                                # NOTE np.argsort with `asc` order，then it converts to `desc` order with [::-1]
        target_ids = np.argsort(scores_column_)[::-1]
        if len(target_ids) > 1000:
            target_ids = target_ids[:1000]
        for i in target_ids:
            mesh_name = shape_names_[i]
            distance = '{:.6f}'.format(1 / scores_column_[i])
            fout.write(mesh_name + ' ' + distance + '\n')








