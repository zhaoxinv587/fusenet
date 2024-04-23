import torch.utils.data
from PIL import Image
from torchvision import transforms
import glob
import os
import numpy as np
import torch
from torch.utils.data import DistributedSampler, Dataset


class ShapeNetCore55_MultiView(torch.utils.data.Dataset):
    def __init__(self, root_dir='data/shrec17', label_file='train.csv',
                 version='normal', num_views=20, total_num_views=20, num_classes=55):
        assert num_classes in [55, ], '`num_classes` should be chosen in this list: [55,]'
        if num_classes == 55:
            self.classnames = list()
            self.shape2class = dict()
            with open(os.path.join(root_dir, label_file)) as fin:
                lines = fin.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue  # skip csv header line
                    if label_file == 'train.csv':
                        shape_name, class_name, _ = line.strip().split(',')
                    # test
                    if label_file == 'test.csv':
                        shape_name, class_name, _, _, _ = line.strip().split(',')
                    if shape_name not in self.shape2class.keys():
                        self.shape2class[shape_name] = class_name
                    if class_name not in self.classnames:
                        self.classnames.append(class_name)
            # it is necessary to sort `self.classnames`, ensuring `classnames` in order in train/test/val
            self.classnames = sorted(self.classnames)

        mode = os.path.splitext(label_file)[0]
        self.root_dir = root_dir
        # e.g, work_dir = 'data/shrec17/train_normal'
        self.work_dir = os.path.join(self.root_dir, f'{mode}_{version}')

        self.num_views = num_views
        self.filepaths = []

        # sorted 这个方法很巧妙
        all_files = sorted(glob.glob(f'{self.work_dir}/*.png'))
        self.filepaths.extend(all_files)

        # NOTE `total_num_views` depends on the dataset where each 3D object corresponds maximum number of views
        #   `num_views` <= `total_num_views`, we can vary `num_views` to conduct ablation studies

        if mode == 'train':
            # 训练模式，打乱顺序
            rand_idx = list(range(len(self.filepaths) // total_num_views))
        else:
            # 验证/测试模式，不打乱顺序
            rand_idx = list(range(len(self.filepaths) // total_num_views))

        filepaths_shuffled = []
        for i in range(len(rand_idx)):
            idx = rand_idx[i]
            start = idx * total_num_views
            end = (idx + 1) * total_num_views
            filepaths_interval = self.filepaths[start:end]
            # NOTE randomly select `num_view` views from `filepaths_interval`
            #   use `np.random.choice`, it is necessary to set `replace=False`
            selected_filepaths = np.random.choice(filepaths_interval, size=(num_views,), replace=False)
            filepaths_shuffled.extend(selected_filepaths)
        self.filepaths = filepaths_shuffled

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        # e.g. 'data/shrec17/val_normal/019180_003.png'
        path = self.filepaths[idx * self.num_views]
        # e.g. '019180_003.png'
        img_name = path.split('/')[-1]
        # e.g. '019180'
        shape_name = img_name[:6]
        # e.g. '04256520'
        class_name = self.shape2class[shape_name]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx * self.num_views + i])
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return class_id, torch.stack(imgs), shape_name

#self.filepaths[idx * self.num_views:(idx + 1) * self.num_views]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
class ShapeNetCore55_Point(torch.utils.data.Dataset):
    def __init__(self, root_dir='/data/shrec17', label_file='train.csv', version='normal', num_classes=55):
        assert num_classes in [55,], '`num_classes` should be chosen in this list: [55,]'
        if num_classes == 55:
            self.classnames = list()
            self.shape2class = dict()
            with open(os.path.join(root_dir, label_file)) as fin:
                lines = fin.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue    # skip csv header line
                    if label_file == 'train.csv':
                        shape_name, class_name, _ = line.strip().split(',')
                    # test
                    if label_file == 'test.csv':
                        shape_name, class_name, _, _, _ = line.strip().split(',')
                    if shape_name not in self.shape2class.keys():
                        self.shape2class[shape_name] = class_name
                    if class_name not in self.classnames:
                        self.classnames.append(class_name)
            # it is necessary to sort `self.classnames`, ensuring `classnames` in order in train/test/val
            self.classnames = sorted(self.classnames)
        mode = os.path.splitext(label_file)[0]
        self.root_dir = root_dir
        # e.g, work_dir = 'data/shrec17/train_normal'
        self.work_dir = os.path.join(self.root_dir, f'{mode}_{version}')
        self.filepaths = []

        all_files = sorted(glob.glob(f'{self.work_dir}/*.obj'))
        self.filepaths.extend(all_files)
        #print(self.filepaths)




    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        # e.g. 'data/shrec17/val_normal/019180_003.png'
        path = self.filepaths[idx]
        # e.g. '019180_003.png'
        img_name = path.split('/')[-1]
        # e.g. '019180'
        shape_name = img_name[:6]
        # e.g. '04256520'
        class_name = self.shape2class[shape_name]
        class_id = self.classnames.index(class_name)
        with open(self.filepaths[idx]) as file:
            #print(self.filepaths[idx])
            points = []
            lines = file.readlines()
            #print('1',lines)
            for line in enumerate(lines):
                line = list(line)[1]
                # print('1',line[1])
                # print('2',line[2])
                strs = line.split(" ")
                #print(strs)
                if line[0] == 'v':
                    points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == 'vt':
                    break
        # points原本为列表，需要转变为矩阵，方便处理
        points = np.array(points)
        point_set = points[:, 0:3]
        seg = points[:, -1].astype(np.int32)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        choice = np.random.choice(len(seg), 1024, replace=True)
        point_set = point_set[choice, :]
        return class_id, point_set ,shape_name


class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        return x1, x2

    def __len__(self):
        return len(self.dataset1)
