from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from sklearn.model_selection import KFold

import cv2

# change DATASET ROOT to your supervised dataset path
DATASET_ROOT = '/home/liu/Downloads/supervised/Agriculture-Vision-2021/' # original 2021 dataset for supervised track

# change DATASET ROOT to your semi-supervised dataset path and manually merge the test image and pseudo gt to your training set
# DATASET_ROOT = '/home/liu/Downloads/Agriculture-Vision-2021_mix_self/' # self-training for semi-supervised track


TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
TEST_ROOT = os.path.join(DATASET_ROOT, 'test/images')


"""
In the loaded numpy array, only 0-8 integer labels are allowed, and they represent the annotations in the following way:

0 - background
1 - double_plant
2 - drydown
3 - endrow
4 - nutrient_deficiency
5 - planter_skip
6 - water
7 - waterway
8 - weed_cluster

"""

# customised palette for visualization, easier for reading in paper
palette_vsl = {
    0: (65, 65, 65),
    1: (224,255,255),
    2: (210,180,140),
    3: (124,252,0),
    4: (255,0,0),
    5: (211,211,211),
    6: (255,215,0),
    7: (255,0, 255),
    8: (60,179,113),
    9: (0,255,0)
}


labels_folder = {
    'endrow': 3,
    'planter_skip': 5,
    'weed_cluster': 8,
    'double_plant': 1,
    'nutrient_deficiency': 4,
    'drydown': 2,
    'waterway':7,
    'water': 6,
    'storm_damage': 0, # 255 ignored class
}



land_classes = ["background", "double_plant", "drydown", "endrow",
                "nutrient_deficiency", "planter_skip", "water",
                "waterway","weed_cluster"
                ]


Data_Folder = {
    'Agriculture2021': {
        'ROOT': DATASET_ROOT,
        'RGB': 'images/rgb/{}.jpg',
        'NIR': 'images/nir/{}.jpg',
        'SHAPE': (512, 512),
        'GT_tr': 'gt/{}.png',
        'GT_val': 'gt/{}.png',
        'GT_tst': 'gt/{}.png',
    },
}


IMG = 'images' # RGB or IRRG, rgb/nir
GT = 'gt'
IDS = 'IDs'


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.png','.jpg'])


def prepare_gt(root_folder = TRAIN_ROOT, out_path='gt'):
    if not os.path.exists(os.path.join(root_folder, out_path)):
        print('----------creating groundtruth data for training./.val---------------')
        check_mkdir(os.path.join(root_folder, out_path))
        basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder,'images/rgb'))]
        gt = basname[0]+'.png'
        for fname in basname:
            gtz = np.zeros((512, 512), dtype=int)
            for key in labels_folder.keys():
                gt = fname + '.png'
                mask = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, gt), -1), dtype=int)
                mask[mask==255] = labels_folder[key]
                gtz[gtz<1] = mask[gtz<1]

            for key in ['boundaries', 'masks']:
                mask = np.array(cv2.imread(os.path.join(root_folder, key, gt), -1), dtype=int)
                gtz[mask == 0] = 255

            cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)


def get_training_list(root_folder = TRAIN_ROOT,
                      gt_folder = Data_Folder['Agriculture2021']['GT_tr'][0:-7],
                      count_label=True):
    dict_list = {}
    basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, gt_folder))]
    if count_label:
        for key in labels_folder.keys():
            no_zero_files=[]
            for fname in basname:
                gt = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, fname+'.png'), -1))
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basname


def split_train_val_test_sets(data_folder=Data_Folder, name='Agriculture2021', bands=['RGB', 'NIR'],
                              KF=6, k=1, seeds=69278, non_val=False):

    train_id, t_list = get_training_list(root_folder=TRAIN_ROOT,
                                         gt_folder = Data_Folder[name]['GT_tr'][0:-7],
                                         count_label=False)
    val_id, v_list = get_training_list(root_folder=VAL_ROOT,
                                       gt_folder = Data_Folder[name]['GT_val'][0:-7],
                                       count_label=False)
    # test_id, ts_list = get_training_list2(root_folder=os.path.join(DATASET_ROOT, 'test'), count_label=False)

    if KF >=2:
        vs_list = [s.split('_') for s in v_list]
        fields = {}
        for i in range(len(vs_list)):
            if vs_list[i][0] not in fields.keys():
                fields[vs_list[i][0]] = list()
            fields[vs_list[i][0]].append(vs_list[i][0] + '_' + vs_list[i][1])

        val_fields = np.array([k for k in fields.keys()])

        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)
        val_ids = np.array(val_fields)
        idx = list(kf.split(np.array(val_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0

        t2_list, v_list = [], []
        for key in val_ids[idx[k][1]]:
            v_list += fields[key]
        for key in val_ids[idx[k][0]]:
            t2_list += fields[key]

        t2_list, v_list = list(t2_list), list(v_list)
        if non_val:
            t2_list = t2_list + v_list

    else:
        if non_val:
            t2_list = v_list
        else:
            t2_list = []


    img_folders = [os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name][band]) for band in bands]
    gt_folder = os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name]['GT_tr'])

    val_folders = [os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name][band]) for band in bands]
    val_gt_folder = os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT_val'])

    # fake for test
    tst_folders = [os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name][band]) for band in bands]
    tst_gt_folder = os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name]['GT_tst'])
    #                    {}
    train_id['all']=t_list+t2_list
    train_dict = {
        IDS: train_id,
        IMG: [[img_folder.format(id) for img_folder in img_folders] for id in t_list] +
             [[val_folder.format(id) for val_folder in val_folders] for id in t2_list] , #+
             # [[tst_folder.format(id) for tst_folder in tst_fol?ders] for id in ts_list], #self_train
        GT: [gt_folder.format(id) for id in t_list] +
            [val_gt_folder.format(id) for id in t2_list] , #+
            # [tst_gt_folder.format(id) for id in ts_list], # self_train
        'all_files': t_list + t2_list #+ ts_list
    }
    val_id['all'] = v_list
    val_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
        'all_files': v_list
    }

    # fake code, for test set, not used
    test_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
    }

    print('train set -------', len(train_dict[GT]))
    print('val set ---------', len(val_dict[GT]))
    return train_dict, val_dict, test_dict



def get_real_test_list(root_folder = TEST_ROOT, data_folder=Data_Folder, name='Agriculture2021', bands=['RGB']):
    dict_list = {}
    basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'nir'))]
    dict_list['all'] = basname[10700:19708]

    test_dict = {
        IDS: dict_list,
        IMG: [os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name][band]) for band in bands],
    }
    return test_dict
