# from tools.model import *
from tools.agri_models import *
from configs.config_agri_v2 import *

# mobilenet backbone, light model
net1_cfg = agriculture2021_configs(
    net_name='PAGNet_mobile',
    data='Agriculture2021',
    bands_list=['RGB', 'NIR'],
    note='testing'
)

# resnext50 backbone, large model
net2_cfg = agriculture2021_configs(
    net_name='PAGNet_rx50',
    data='Agriculture2021',
    bands_list=['RGB', 'NIR'],
    note='testing'
)


# ------------ supervised trained models -----------------------------------#
net1_k2 = 'epoch_42_loss_1.68048_acc_0.80724_acc-cls_0.67384_mean-iu_0.54108_fwavacc_0.68180_f1_0.69454_.pth'  # two stage inceasing batch size train 12-18
net1_k4 = 'epoch_35_loss_1.67883_acc_0.82601_acc-cls_0.72034_mean-iu_0.58952_fwavacc_0.71319_f1_0.73355_.pth'  # three overfitting train
net1_k5 = 'epoch_22_loss_1.72611_acc_0.80056_acc-cls_0.67588_mean-iu_0.53288_fwavacc_0.67224_f1_0.68128_.pth'  # one stage train

net2_k2 = 'epoch_29_loss_1.78271_acc_0.77234_acc-cls_0.59314_mean-iu_0.44354_fwavacc_0.63742_f1_0.60233_.pth'
net2_k4 = 'epoch_32_loss_1.75012_acc_0.79587_acc-cls_0.65263_mean-iu_0.53648_fwavacc_0.66824_f1_0.68798_.pth'
net2_k5 = 'epoch_25_loss_1.71590_acc_0.80142_acc-cls_0.67413_mean-iu_0.54713_fwavacc_0.67079_f1_0.69418_.pth'


# -------------- semi-supervised model with noise label and overfitting training ------ #
net1_k4_s512 = 'epoch_27_loss_1.71430_acc_0.81710_acc-cls_0.71291_mean-iu_0.56484_fwavacc_0.70791_f1_0.71135_.pth' # trained wtih full size 512x512
net2_k5_s384 = 'epoch_31_loss_1.65110_acc_0.82508_acc-cls_0.73910_mean-iu_0.61429_fwavacc_0.70495_f1_0.75122_.pth' # IoU 516


def get_net(ckpt=net1_cfg):
    net = load_model(name=ckpt.model, classes=ckpt.nb_classes,
               m_views=ckpt.m_views, kernel=ckpt.kernel, hidden_ch=ckpt.hidden_ch,
               heads=ckpt.heads, depth=ckpt.depth,
               dropout=ckpt.dropout, activation=ckpt.activation,
                     shortcut=ckpt.shortcut, ndvi=ckpt.ndvi,
            )

    return net


def loadtestimg(test_files):
    id_dict = test_files[IDS]
    image_files = test_files[IMG]
    # mask_files = test_files[GT]

    for key in id_dict.keys():
        for id in id_dict[key]:
            if len(image_files) > 1:
                imgs = []
                for i in range(len(image_files)):
                    filename = image_files[i].format(id)
                    path, _ = os.path.split(filename)
                    if path[-3:] == 'nir':
                        img = np.asarray(Image.open(filename), dtype='uint8')
                        img = np.expand_dims(img, 2)

                        imgs.append(img)
                    else:
                        img = imload(filename)
                        imgs.append(img)
                image = np.concatenate(imgs, 2)
            else:
                filename = image_files[0].format(id)
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    image = np.asarray(Image.open(filename), dtype='uint8')
                    image = np.expand_dims(image, 2)
                else:
                    image = imload(filename)

            yield image


def loadids(test_files):
    id_dict = test_files[IDS]

    for key in id_dict.keys():
        for id in id_dict[key]:
            yield id


def loadgt(test_files):
    id_dict = test_files[IDS]
    mask_files = test_files[GT]
    for key in id_dict.keys():
        for id in id_dict[key]:
            label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')
            yield label
