
from torchvision import transforms, datasets
import torch.utils.data
import cv2
import glob
from PIL import Image
#class TIMITDataset(Dataset)创建dataset类。里面定义了三个函数，def __init__(), def __getitem__(), def __len__(self)（相当于重写了dataset类）。

class ImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False,
                 num_models=0):
        # self.classnames=[ 'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox' ]

        self.classnames=['bathtub','bed','chair','desk','dresser','monitor','night_stand',
                         'sofa','table','toilet']
                         # airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         # #                                                                           'cone', 'cup',
                         # # 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                         # # 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                         # # 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                         # # 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png'))

            self.filepaths.extend(all_files)   #filepath class <list>

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            #transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        im = Image.open(self.filepaths[idx])
        # im_depth = cv2.imread(self.filepaths[idx])
        # if self.classnames[3] == 'bench' \
        #         or self.classnames[6] == 'bowl' \
        #         or self.classnames[10] == 'cup' \
        #         or self.classnames[12] == 'desk'\
        #         or self.classnames[14] == 'dresser'\
        #         or self.classnames[15] == 'flower_pot' \
        #         or self.classnames[19] == 'lamp' \
        #         or self.classnames[23] == 'night_stand' \
        #         or self.classnames[25] == 'piano' \
        #         or self.classnames[26] == 'plant' \
        #         or self.classnames[27] == 'radio' \
        #         or self.classnames[29] == 'sink' \
        #         or self.classnames[31] == 'stairs' \
        #         or self.classnames[32] == 'stool' \
        #         or self.classnames[33] == 'table' \
        #         or self.classnames[36] == 'tv_stand' \
        #         or self.classnames[37] == 'vase' \
        #         or self.classnames[38] == 'wardrobe' \
        #         or self.classnames[39] == 'xbox' \
        #         :
        #     im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=1), cv2.COLORMAP_JET)
        #     im = Image.fromarray(im_color)
        # else:
        #     im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (im,class_id)