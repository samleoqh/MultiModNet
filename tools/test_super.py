from __future__ import division

from lib.visual import *
from lib.functions import *
import torchvision.transforms as st
from sklearn.metrics import confusion_matrix
from tqdm import tqdm_notebook as tqdm

from tools.agri_ckpt_best import *
import time
import itertools

ckpt_root_semi = '../ckpt/supervise_trained'

output_folder = os.path.join('../test_output', 'supervised')

def main():
    check_mkdir(output_folder)

    net1_ckpt_list = [
        net1_k2,
        net1_k4,
        net1_k5,
    ]

    net2_ckpt_list = [
        net2_k2,
        net2_k4,
        net2_k5,
    ]

    # resize input based on trained model
    scale_list = [
        384. / 512., # for net1
        384. / 512., # for net1
        384. / 512., # for net1
        384. / 512., # for net2
        384. / 512., # for net2
        384. / 512., # for net2
        # 512. / 512.
                   ]
    nets = []
    for ckpt in net1_ckpt_list:
        net = get_net(net1_cfg)
        net.load_state_dict(torch.load(os.path.join(ckpt_root_semi, ckpt)))
        net.cuda()
        net.eval()
        nets.append(net)

    for ckpt in net2_ckpt_list:
        net = get_net(net2_cfg)
        net.load_state_dict(torch.load(os.path.join(ckpt_root_semi, ckpt)))
        net.cuda()
        net.eval()
        nets.append(net)

    test_files = get_real_test_list(bands=['RGB','NIR'])
    tta_real_test(nets, stride=512, batch_size=1, scales=scale_list,
                  norm=False, window_size=(384, 384), labels=land_classes,
                  test_set=test_files, all=True)


def fusion_tta(nets, image, scales, batch_size=1, num_class=7, wsize = (512, 512)):
    pred_all = np.zeros(image.shape[:2] + (num_class,))
    sharpen_temp = len(scales)>1
    for net, scale_rate in zip(nets, scales):
        # print('scale rate: ', scale_rate)
        img = image.copy()
        img = scale(img, scale_rate)
        pred = np.zeros(img.shape[:2] + (num_class,))
        stride = img.shape[1]
        window_size = img.shape[:2]
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):

            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]

            imgs_flip = [patch[:, ::-1, :] for patch in image_patches]
            imgs_mirror = [patch[:, :, ::-1] for patch in image_patches]
            imgs_rot90 = [torch_rot90_(torch.from_numpy(patch)).numpy() for patch in image_patches]
            imgs_rot180 = [torch_rot180_(torch.from_numpy(patch)).numpy() for patch in image_patches]

            image_patches = np.concatenate(
                (image_patches, imgs_flip, imgs_mirror, imgs_rot180, imgs_rot90), axis=0)
            image_patches = np.asarray(image_patches)
            image_patches = torch.from_numpy(image_patches).cuda()

            outs = net(image_patches) # + torch_rot270(net(torch_rot90(image_patches)))

            outs = torch.softmax(outs, 1)
            outs = (outs**0.5)/5.0  # shappen v0
            outs = outs.data.cpu().numpy()

            b, _, _, _ = outs.shape

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs[0:b // 5, :, :, :], coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out

            for out, (x, y, w, h) in zip(
                    outs[b // 5:2 * b // 5, :, :, :], coords):
                out = out[:, ::-1, :]  # flip back
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out

            for out, (x, y, w, h) in zip(
                    outs[2 * b // 5: 3 * b // 5, :, :, :], coords):
                out = out[:, :, ::-1]  # mirror back
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out

            for out, (x, y, w, h) in zip(
                    outs[3 * b // 5: 4 * b // 5, :, :, :], coords):
                out = torch_rot180_(torch.from_numpy(out)).numpy()
                # out = torch_rot270_(torch.from_numpy(out)).numpy()
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out

            for out, (x, y, w, h) in zip(
                    outs[4 * b // 5: b, :, :, :], coords):
                out = torch_rot270_(torch.from_numpy(out)).numpy()
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out

            del (outs)

        if sharpen_temp:
            pred_all += scale(pred, 1.0 / scale_rate) ** 0.5
        else:
            pred_all += scale(pred, 1.0 / scale_rate)

    return pred_all


def tta_real_test(nets, all=False, labels=land_classes, norm=False,scales=[1.0],
             test_set=None, stride=600, batch_size=5, window_size=(512, 512)):
    # test_images, test_labels = (loadtestimg(test_set))
    test_files = (loadtestimg(test_set))
    # gts = (loadgt(test_set))
    idlist=(loadids(test_set))

    all_preds = []
    # all_gts = []
    num_class = len(labels)
    ids = []

    total_ids = 0

    for k in test_set[IDS].keys():
        total_ids += len(test_set[IDS][k])

    for img, id in tqdm(zip(test_files, idlist), total=total_ids, leave=False):
        # org_img = img.copy()  # for visualization
        img = np.asarray(img, dtype='float32')
        img = st.ToTensor()(img)
        img = img / 255.0
        if norm:
            img = st.Normalize(*mean_std)(img)
        img = img.cpu().numpy().transpose((1, 2, 0))

        stime = time.time()

        with torch.no_grad():
            pred = fusion_tta(nets, image=img, scales=scales,
                              batch_size=batch_size, num_class=num_class, wsize=window_size)

        print('inference cost time: ', time.time() - stime)

        pred = np.argmax(pred, axis=-1)

        for key in ['boundaries', 'masks']:
            pred = pred * np.array(cv2.imread(os.path.join(DATASET_ROOT, 'test', key, id+'.png'), -1) / 255, dtype=int)
        filename = './{}.png'.format(id)
        cv2.imwrite(os.path.join(output_folder, filename), pred)
        all_preds.append(pred)
        ids.append(id)

    if all:
        return all_preds, ids
    else:
        return all_preds


def metrics(predictions, gts, label_values=land_classes):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except BaseException:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    return accuracy, cm


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    # print(top.shape[0])
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    print('total number of sliding windows: ', c)
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()
