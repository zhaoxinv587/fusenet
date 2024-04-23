from FuseNet.models.FuseNet_ModelNet40 import FuseNet ,FC,W,W1
#from modelnet40dataset import MultiviewImgDataset , ModelNetDataLoader
from FuseNet.mdata import MultiviewImgDataset , ModelNetDataLoader
import argparse
import logging
from tqdm import tqdm
import sys
import importlib
import torch
import os
import torch.utils.data as DATA
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt

def get_AP_values(all_true, all_pred, class_num=40):
    # 将所有类别转化为onehot标签
    all_true = label_binarize(all_true, classes=list(range(class_num)))
    # print(all_true)
    # print(all_pred)
    # all_pred.sort()
    # 统计每个类别的AP值
    precision = dict()
    recall = dict()
    average_precision = dict()
    kaka = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
            'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
            'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    ap=0
    for i in range(class_num):
        precision[i], recall[i], _ = precision_recall_curve(all_true[:, i],
                                                            all_pred[:, i])

        average_precision[i] = average_precision_score(all_true[:, i],
                                                       all_pred[:, i])
        ap+=average_precision[i]
    print('mAP',ap/40)

    # 统计宏均值AP
    precision["macro"], recall["macro"], _ = precision_recall_curve(all_true.ravel(),
                                                                    all_pred.ravel())

    average_precision["macro"] = average_precision_score(all_true, all_pred,
                                                         average="macro")
    average_precision["micro"] = average_precision_score(all_true, all_pred,
                                                         average="micro")
    # 打印结果
    #print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision["macro"]))

    # 转换格式
    average_precision_df = pd.DataFrame([average_precision]).T
    average_precision_df.index = kaka + ['macro'] +['micro']
    average_precision_df.to_csv('./all_AP_values.csv', encoding='utf_8_sig')

    # 可视化
    plt.figure(figsize=(12, 8))
    # for i in range(class_num):
    #     plt.step(recall[i], precision[i], where='post')

    plt.step(recall['macro'], precision['macro'], where='pre')
    #plt.step(recall['airplane'], precision['airplane'], where='pre')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    # plt.title(
    #     'Average precision score, mean-averaged over all classes: AP={0:0.3f}'.format(average_precision["macro"]))
    plt.title(
        'Average precision score, mean-averaged over all classes: AP = 0.9404')
    plt.show()
    return average_precision_df





def get_precision_and_recall(y_label, y_prob):
    """

    :param y_label: true label
    :param y_score: predict probability
    :param n:
    :return:
    """
    assert len(y_label) == len(y_prob)
    # invert sort y_pred
    score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_prob = np.array(y_prob)[score_indices]
    y_true = np.array(y_label)[score_indices]

    # ------------------get tps and fps at distinct value -------------------------
    # extract the indices associated with the distinct values
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
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


def pr_plot(precision, recall, area):

    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, linestyle='-', linewidth=2,
             label='Precision-Recall Curve Area={:.4f}'.format(area))
    plt.fill_between(recall, precision, color='C0', alpha=0.4, interpolate=True)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=15)
    plt.legend(loc="upper right")
    plt.show()



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../models'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='Pointnet_modelnet40',help='Experiment root') #pointnet2_cls_msg
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
    data_path = '../data/modelnet40_images_points/modelnet40_normal_resampled/'
    val_Pointsdataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    val_PointsLoader = torch.utils.data.DataLoader(val_Pointsdataset, batch_size=1, shuffle=False, num_workers=10)

    #多视图
    testpath = "data/modelnet40_images_12views/*/test"
    val_Mutiviewdataset = MultiviewImgDataset(testpath)
    val_Mutiviewloader = torch.utils.data.DataLoader(val_Mutiviewdataset, batch_size=1, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    #点云
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    model1= model.get_model(num_class, normal_channel=args.use_normals)
    model1 = model1.to(device)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    model1.load_state_dict(checkpoint['model_state_dict'])

    #多视图
    model2 = FuseNet().to(device)
    model2.load_state_dict(torch.load('../log/Mutiview/FuseNet3.pth'))

    #FC
    modelFC = FC().to(device)
    modelFC.load_state_dict(torch.load('../log/FC/FC_pointnet2.pth'))

    #权重W
    modelW = W().to(device)
    modelW.load_state_dict(torch.load('../log/W/W_pointnet2.pth'))

    #权重W1
    modelW1 = W1().to(device)
    modelW1.load_state_dict(torch.load('../log/W1/W1_pointnet2.pth'))



    '''Eval'''

    print('num_test_files(Multiview): ' + str((len(val_Mutiviewdataset.filepaths))/12))

    model1.eval()
    model2.eval()  # ，model.eval()是保证BN用全部训练数据的均值和方差
    modelFC.eval()
    modelW.eval()
    modelW1.eval()
    all_true = []
    all_pred = []
    #gg = []
    correct, total = 0, 0
    for (labels, inputs), (j, (points, target)) in zip(val_Mutiviewloader , tqdm(enumerate(val_PointsLoader), total=len(val_PointsLoader))) :
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
        #gg.append(test)
        true_label = labels.cpu().numpy().tolist()
        pred_label = torch.nn.functional.softmax(fuse, dim=1).cpu().detach().numpy().tolist()
        all_true += true_label
        all_pred += pred_label



        correct += torch.eq(pred, labels).sum().item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))
    #nextVal = sorted(gg, reverse=True)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    # precision, recall, ap = get_precision_and_recall(all_true,all_pred)
    # print(precision)
    # print(recall)
    # print(ap/40)
    # pr_plot(precision, recall, ap)
    average_precision_df = get_AP_values(all_true, all_pred)
    print(average_precision_df)
    #average_precision_df



if __name__ == '__main__':
    args = parse_args()
    main(args)




