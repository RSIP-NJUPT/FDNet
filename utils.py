import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image


def normalize(x):
    norm_x = np.zeros(x.shape)
    for i in range(x.shape[2]):
        input2_max = np.max(x[:, :, i])
        input2_min = np.min(x[:, :, i])
        norm_x[:, :, i] = (x[:, :, i] - input2_min) / (input2_max - input2_min)

    return norm_x


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def Accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def trPixel2Patch(Data1, Data2, patchsize, pad_width, TR_Label):
    """
        Data1: hsi
        Data2: lidar
    """
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)

    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2

    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2

    [ind1, ind2] = np.where(TR_Label > 0)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    TrainNum = len(ind1)

    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)

    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1

        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2

        patchlabel = TR_Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)

    # add dimension for 3D-CNN
    # TrainPatch1 = TrainPatch1.unsqueeze(1)

    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel


def tsPixel2Patch(Data1, Data2, patchsize, pad_width, TS_Label):
    """
        Data1: hsi
        Data2: lidar
    """
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)

    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2

    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2

    [ind1, ind2] = np.where(TS_Label > 0)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    TestNum = len(ind1)

    TestPatch1 = np.empty((TestNum, l1, patchsize, patchsize), dtype='float32')
    TestPatch2 = np.empty((TestNum, l2, patchsize, patchsize), dtype='float32')
    TestLabel = np.empty(TestNum)

    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TestPatch1[i, :, :, :] = patch1

        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TestPatch2[i, :, :, :] = patch2

        patchlabel = TS_Label[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    TestPatch1 = torch.from_numpy(TestPatch1)
    TestPatch2 = torch.from_numpy(TestPatch2)

    # add dimension
    # TestPatch1 = TestPatch1.unsqueeze(1)

    TestLabel = torch.from_numpy(TestLabel) - 1
    TestLabel = TestLabel.long()
    return TestPatch1, TestPatch2, TestLabel, ind1, ind2


def train_epoch(model, train_loader, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])

    for batch_idx, (batch_data1, batch_data2, batch_target) in enumerate(train_loader):
        batch_data1 = batch_data1.cuda()
        batch_data2 = batch_data2.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data1, batch_data2)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = Accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, valid_loader, criterion, get_ts_result):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    # add
    test_result = []

    for batch_idx, (batch_data1, batch_data2, batch_target) in enumerate(valid_loader):
        batch_data1 = batch_data1.cuda()
        batch_data2 = batch_data2.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data1, batch_data2)
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = Accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

        # add
        _, batch_pred_result = batch_pred.max(1)
        test_result.extend((batch_pred_result + 1).cpu().numpy())

    if get_ts_result:
        return test_result

    return tar, pre


def draw_classification_map(label_matrix, img_path: str, dataset_name: str = 'Houston'):
    if dataset_name == 'Houston':
        color_map = {
            0: (0, 0, 0),
            1: (219, 94, 86),  # Healthy grass
            2: (219, 147, 86),  # Stressed grass
            3: (219, 200, 86),  # Synthetic grass
            4: (184, 219, 86),  # Tree
            5: (131, 219, 86),  # Soil
            6: (86, 219, 94),  # Water
            7: (86, 219, 147),  # Residential
            8: (86, 219, 200),  # Commercial
            9: (86, 184, 219),  # Road
            10: (85, 130, 217),  # Highway
            11: (93, 85, 216),  # Railway
            12: (145, 85, 216),  # Park lot 1
            13: (195, 84, 214),  # Park lot 2
            14: (216, 85, 182),  # Tennis court
            15: (216, 85, 129),  # Running track
        }
    elif dataset_name == 'Trento':
        color_map = {
            0: (0, 0, 0),
            1: (56, 86, 159),  # apple trees
            2: (81, 204, 237),  # Buildings
            3: (156, 206, 109),  # Ground
            4: (251, 209, 26),  # Woods
            5: (236, 53, 37),  # Vineyard
            6: (125, 21, 22),  # Roads
        }
    elif dataset_name == 'Augsburg':
        color_map = {
            0: (0, 0, 0),
            1: (56, 108, 52),  # Forest
            2: (228, 55, 38),  # Residential Area
            3: (229, 240, 84),  # Industrial Area
            4: (170, 228, 49),  # Low Plants
            5: (139, 222, 38),  # Allotment
            6: (173, 238, 235),  # Commercial Area
            7: (85, 132, 192),  # Water
        }
    elif dataset_name == 'Muufl':
        color_map = {
            -1: (0, 0, 0),
            1: (59, 132, 70),  # Trees
            2: (83, 172, 71),  # Mostly grass
            3: (0, 204, 204),  # Mixed ground surface
            4: (146, 82, 52),  # Dirt and sand
            5: (218, 50, 43),  # Road
            6: (103, 189, 199),  # Water
            7: (229, 229, 240),  # Buildings shadow
            8: (199, 177, 202),  # Buildings
            9: (218, 142, 51),  # Sidewalk
            10: (224, 220, 83),  # Yellow curb
            11: (228, 119, 90),  # Cloth panels
        }
    elif dataset_name == 'Dafeng':
        color_map = {
            1: (59, 132, 70),  # 碱蓬
            2: (0, 204, 204),  # 裸土
            3: (146, 82, 52),  # 道路
            4: (218, 50, 43),  # 房屋
            5: (125, 21, 22),  # 水体
            6: (56, 86, 159),  # 芦苇
            7: (251, 209, 26),  # 互花米草
            8: (145, 85, 216),  # 藻类
            9: (81, 204, 237),  # 草类
        }
    else:
        raise 'datasets name error'

    label_matrix = np.array(label_matrix).astype(np.uint8)
    rgb_array = np.zeros((label_matrix.shape[0], label_matrix.shape[1], 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        rgb_array[label_matrix == class_id] = color

    rgb_image = Image.fromarray(rgb_array)
    rgb_image.save(img_path)
