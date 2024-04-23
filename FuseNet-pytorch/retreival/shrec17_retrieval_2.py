import os

import numpy as np
from torch import nn, optim
from scipy.special import softmax
from tqdm import tqdm
from torch.autograd import Variable
from FuseNet2.shrec17.shapenetcore55dataset import MyDataset
from FuseNet2.shrec17.Vgg16 import VGGNet16
from FuseNet2.shrec17.shapenetcore55dataset import ShapeNetCore55_Point , ShapeNetCore55_MultiView
import torch
from FuseNet2.points_extraction import pointnet2_cls as pointnet
from shrec17.fusenet2_shrec17 import W1,FuseNet2

# Check cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    #points
    test_Pointsdataset= ShapeNetCore55_Point(root_dir='/home/ma2/zx/shrec17_trained_feature', label_file='test.csv', version='normal', num_classes=55)

    #images
    test_Mutiviewdataset = ShapeNetCore55_MultiView(root_dir='/media/ma2/3EEF1FF3BAE5C446/data/shrec17_points/shrec17_images_20views', label_file='test.csv',
                                                    version='normal', num_views=20, total_num_views=20, num_classes=55)


    #images and points

    testDataset = MyDataset(dataset1=test_Mutiviewdataset, dataset2=test_Pointsdataset)
    testdataloader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=1, shuffle=False,drop_last=True)

    #images model
    modelimages = VGGNet16().to(device)
    modelimages.load_state_dict(torch.load('/home/ma2/zx/FuseNet2/log/Shrec17_vgg16_20views.pth'))
    modelimages.eval()


    #FuseNet2
    modelFuseNet = FuseNet2().cuda()
    modelFuseNet.load_state_dict(torch.load('/home/ma2/zx/FuseNet2/shrec17/FuseNet2_shrec17.pth'))
    optFuseNet = optim.SGD(modelFuseNet.parameters(),lr=0.01,momentum=0.8,weight_decay=0.001)
    schedulerFuseNet = optim.lr_scheduler.ExponentialLR(optFuseNet,gamma=0.96)

    W1
    modelW1 = W1().cuda()
    modelW1.load_state_dict(torch.load('/home/ma2/zx/FuseNet2/shrec17/W1_shrec17.pth'))
    optW1 = optim.SGD(modelW1.parameters(), lr=0.01, momentum=0.8, weight_decay=0.001)
    schedulerW1 = optim.lr_scheduler.ExponentialLR(optW1, gamma=0.96)



    """Eval"""

    print('num_test_files(Multiview): ' + str(len(test_Mutiviewdataset.filepaths)/20))
    print('num_test_files(Points): ' + str(len(test_Pointsdataset.filepaths)/20))
    modelFuseNet.eval()
    modelW1.eval()
    correct, total = 0, 0
    pred_logits = []
    pred_class_ids = []
    all_true = []
    all_pred = []
    for _, data in tqdm(enumerate(testdataloader), total=len(testdataloader)):
        with torch.no_grad():
            datapoint = data[1]
            dataimages = data[0]
            inputs = dataimages[1].to(device)
            labels = dataimages[0].to(device)
            inputs = inputs.view(-1, 1, 224, 224)

            points = datapoint[1]
            target1 = datapoint[0]
            in_data = Variable(points)

            outputs = modelimages(inputs)  # logits 全连接层的输出 softmax的输入
            fuse_max = torch.max(torch.cat((in_data,outputs.view(-1,20,512)),2),1)[0]
            fuse_mean = torch.mean(torch.cat((in_data,outputs.view(-1,20,512)),2),1)
            fuse , w1 ,w2 = modelW1(fuse_mean,fuse_max)
            fuse = modelFuseNet.fc2(fuse)
            fuse = modelFuseNet.fc3(fuse).cpu().numpy()

            logits = softmax(fuse, axis=1)
            class_ids = logits.argmax(axis=1)
            pred_logits.extend(logits)
            pred_class_ids.extend(class_ids)


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
filepaths = np.array(test_Mutiviewdataset.filepaths)[::20]  # take one view every `num_views`
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
method_name = 'fusenet2'
savedir = join('evaluator', method_name,f'{mode}_normal')
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
