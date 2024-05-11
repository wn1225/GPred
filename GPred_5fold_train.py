import argparse
from itertools import product
import os

import torch

torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN, Sigmoid
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch_geometric.nn import global_max_pool, radius, global_mean_pool, knn
from torch_geometric.nn.pool import radius
from sklearn import metrics
from PointTransformerConv import PointTransformerConv
import sys
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.spatial
from torch_scatter import scatter_add

Center = T.Center()
Normalscale = T.NormalizeScale()
Delaunay = T.Delaunay()
Normal = T.GenerateMeshNormals()


def normalize_point_pos(pos):
    # pos_AB=torch.cat([pos_A, pos_B])
    pos = pos - pos.mean(dim=-2, keepdim=True)
    # pos_B=pos_B-pos_AB.mean(dim=-2, keepdim=True)
    scale = (1 / pos.abs().max()) * 0.999999
    pos = pos * scale
    # scale_B = (1 / pos_B.abs().max()) * 0.999999
    # pos_B = pos_B * scale_B
    return pos


def load_data(data_path):
    print('loading data')
    data_list = []

    with open(data_path, 'r') as f:
        n_g = int(f.readline().strip())
        num = 0
        for i in range(n_g):  # for each protein
            n = int(f.readline().strip())  # atom number
            point_tag = []
            point_fea_pssm = []
            point_pos = []
            point_aa = []
            aa_y = []
            mask = []
            mask_t = []

            for j in range(n):
                row = f.readline().strip().split()
                point_tag.append(int(row[2])) 
                mask.append(int(row[3])) 
                mask_t.append(int(row[0])) 
                pos, fea_pssm = np.array([float(w) for w in row[4:7]]), np.array([float(w) for w in row[7:]])
                point_pos.append(pos)
                point_fea_pssm.append(fea_pssm)
                point_aa.append(int(row[1])) 

            flag = -1
            for i in range(len(point_aa)):
                if (flag != point_aa[i]):
                    flag = point_aa[i]
                    aa_y.append(point_tag[i])  # label , residue level
            # print(aa_y)
            try:
                x = torch.tensor(point_fea_pssm, dtype=torch.float)  # 59
            except:
                print(x.shape)

            y = torch.tensor(point_tag)
            pos = torch.tensor(point_pos, dtype=torch.float)  # 3
            mask = torch.tensor(mask)
            mask_t = torch.tensor(mask_t)

            # pos=normalize_point_pos(pos)
            data = Data(x=x, y=y, pos=pos)
            # print(data.norm)

            for i in range(len(point_aa)):
                point_aa[i] = point_aa[i] + num
            num = num + len(aa_y)

            aa = torch.tensor(point_aa)
            # print(aa)
            number = len(aa_y)  
            aa_y = torch.tensor(aa_y)

            data.aa = aa
            data.aa_y = aa_y
            data.num = number
            data.mask = mask
            data.mask_t = mask_t

            data = Center(data)
            # data = Normalscale(data)
            data = Delaunay(data)
            data = Normal(data)

            data = data.to(device)
            data_list.append(data)
    # print(data_list)
    return data_list


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU(), Dropout(0.8))
        for i in range(1, len(channels))
    ])


def generate_normal(pos, batch):
    data_norm = []
    batch_list = torch.unique(batch)
    for b in batch_list:
        pos_temp = pos[batch == b]
        pos_temp = pos_temp - pos_temp.mean(dim=-2, keepdim=True)
        pos_temp = pos_temp.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos_temp, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices)

        data_face = face.t().contiguous().to(device, torch.long)
        pos_temp = torch.tensor(pos_temp).to(device)

        vec1 = pos_temp[data_face[1]] - pos_temp[data_face[0]]
        vec2 = pos_temp[data_face[2]] - pos_temp[data_face[0]]
        face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

        idx = torch.cat([data_face[0], data_face[1], data_face[2]], dim=0)
        face_norm = face_norm.repeat(3, 1)

        norm = scatter_add(face_norm, idx, dim=0, dim_size=pos_temp.size(0))
        norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]

        data_norm.append(norm)

    return torch.cat(data_norm, dim=0)


class PointTransformerConv1(torch.nn.Module):
    def __init__(self, r, in_channels, out_channels):
        super(PointTransformerConv1, self).__init__()
        self.k = None
        self.r = r
        self.pos_nn = MLP([6, out_channels])

        self.attn_nn = MLP([out_channels, out_channels])

        self.conv = PointTransformerConv(in_channels, out_channels,
                                         pos_nn=self.pos_nn,
                                         attn_nn=self.attn_nn)

    def forward(self, x, pos, normal, batch):
        # row, col = knn(pos, pos, self.k, batch, batch)
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, edge_index, normal, self.r)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.conv1 = PointTransformerConv1(5, in_channels=39 + 20, out_channels=128)
        self.neck = Seq(Lin(128, 256), BN(256), ReLU(), Dropout(0.3))
        self.mlp = Seq(Lin(256, 128), BN(128), ReLU(), Dropout(0.3), Lin(128, out_channels))

    def forward(self, data, use=None):
        x0, pos, batch, normal, pool_batch, aa_num, mask, mask_t = data.x, data.pos, data.batch, data.norm, data.aa, data.num, data.mask, data.mask_t

        # atom to residue
        flag = torch.Tensor([-1]).to(device)
        num = -1
        for i in range(len(pool_batch)):
            if not torch.eq(pool_batch[i], flag):
                flag = pool_batch[i].clone()
                num = num + 1
                pool_batch[i] = torch.Tensor([num]).to(device)
            else:
                pool_batch[i] = torch.Tensor([num]).to(device)

        x1 = self.conv1(x0, pos, normal, batch)

        out = self.neck(torch.cat([x1], dim=1))
        out = global_max_pool(out, pool_batch)  # out-512
        # print(out)

        # residual batch
        # print(num)
        num_total = 0
        for i in range(len(aa_num)):
            num_total += aa_num[i]
        # print(num_total)
        aa_batch = torch.zeros(num_total).to(device)
        number = 0
        for m in range(len(aa_num)):
            # print(m)
            for n in range(aa_num[m].item()):
                aa_batch[n + number] = m
            number += aa_num[m].item()
        aa_batch = aa_batch.long()

        aa_pos = global_mean_pool(pos, pool_batch)

        aa_norm = generate_normal(aa_pos, aa_batch).to(device)

        # out = self.conv4(out, aa_pos, aa_norm, aa_batch)
        out = self.mlp(out)

     
        mask_t = global_max_pool(mask_t.unsqueeze(dim=1), pool_batch).squeeze()
        mask_t = mask_t == 1

        data.label = data.aa_y[mask_t]
        out = out[mask_t]

        return out


def BCE_loss(inputs, targets, m = 1):
    pos_weight = torch.tensor([m]).to(device)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss(inputs, targets)
    return loss


def train_model(model, patience, n_epochs, checkpoint, m):
    train_losses = []
    valid_losses = []
    label_total = []
    score_total = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, path=checkpoint, verbose=True)

    for epoch in range(1, n_epochs + 1):

        model.train()
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            label = data.label.float()
            label = label.unsqueeze(1)
            # loss = focalloss(out, label, mask_t)
            loss = BCE_loss(out, label, m)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for data in valloader:
                data = data.to(device)
                out = model(data)
                score = torch.sigmoid(out)
                score_total.extend(score.detach().cpu().numpy())
                label = data.label.float()
                label = label.unsqueeze(1)
                label_total.extend(label.detach().cpu().numpy())
                # loss = focalloss(out, label,mask_t)
                loss = BCE_loss(out, label, m)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        auc = metrics.roc_auc_score(label_total, score_total)
        ap = metrics.average_precision_score(label_total, score_total)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'AUC: {auc:.5f}' +
                     f'AP: {ap:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []
        label_total = []
        score_total = []
        print("Learning rate of the %dth epoch：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step(valid_loss)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # model = torch.load(checkpoint)

    return avg_train_losses, avg_valid_losses


parser = argparse.ArgumentParser(
    description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
parser.add_argument('-input', dest='input', type=str, help='Specify the input feature file', required=True)
parser.add_argument('-tnum', dest='tnum', type=str,
                    help='The last 20 percent of the data is used as the test set, specifying the starting position of the test set',
                    required=True)
parser.add_argument('-cpath', dest='cpath', type=str, help='Storage location of model parameter files', required=True)
parser.add_argument('-spath', dest='spath', type=str, help='Storage location for predicted scores', required=True)
parser.add_argument('-mpath', dest='mpath', type=str, help='Storage location for metrics file', required=True)
parser.add_argument('-ion', dest='ion', type=str, help='Specify the ion', required=True)
parser.add_argument('-residus', dest='residus', type=str, help='Specify the ion', required=True)

args = parser.parse_args()

input = args.input
tnum = args.tnum
cpath = args.cpath
spath = args.spath
mpath = args.mpath
ion = args.ion
residus = args.residus

os.makedirs(cpath, exist_ok=True)
os.makedirs(spath, exist_ok=True)
os.makedirs(mpath, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = load_data(input)
folds_t,folds_v = divide_cdhit(dataset, int(tnum))
test = dataset[int(tnum):]
testloader = DataLoader(test, batch_size=4)
directory_name = mpath
file_name = "metrics_no_surface_no_resolution"
file_path = os.path.join(directory_name, file_name)
ff = open(file_path, 'w')


for fold in [0, 1, 2, 3, 4]:

    ff.write(str(fold) + '_model_new_no_surface' + '\n')
    # train, val = divide_cdhit(dataset, fold, int(tnum))
    train = []
    val = []
    for j in folds_t[fold]:
        train.append(dataset[j])
    for i in folds_v[fold]:
        val.append(dataset[i])

    trainloader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
    valloader = DataLoader(val, batch_size=4)

    neg_num, pos_num = weight(ion, residus, folds_t[fold])
    ff.write('neg num:' + str(neg_num) + r'\t' + 'pos num' + str(pos_num) + r'\t' + 'pos weight:' + str(m) + r'\t')

    model = Net()
    # focalloss = FocalLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  #
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    n_epochs = 15000
    patience = 20
    checkpoint = cpath + str(fold) + '_new_no_surface.pt'
    train_loss, valid_loss = train_model(model, patience, n_epochs, checkpoint, 1)

    pred_total = []
    aa_total = []
    out_total = []

    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            out = model(data)
            out = F.sigmoid(out)
            out_total.extend(out.cpu().tolist())
            pred = out.ge(0.5).float()
            pred_total.extend(pred.detach().cpu().numpy())
            aa_total.extend(data.label.detach().cpu().numpy())

    pred_total = torch.tensor(pred_total)
    out_total = torch.tensor(out_total)
    pred_total = pred_total.squeeze()
    out_total = out_total.squeeze()

    aa_total = torch.tensor(aa_total)

    correct = int(pred_total.eq(aa_total).sum().item())
    tn, fp, fn, tp = confusion_matrix(aa_total, pred_total).ravel()
    ff.write('tn' + str(tn) + 'tp' + str(tp) + 'fn' + str(fn) + 'fp' + str(fp) + '\n')

    recall = tp / (tp + fn)
    ff.write('recall:' + str(recall) + '\n')

    sp = tn / (fp + tn)
    ff.write('sp:' + str(sp) + '\n')

    precision = tp / (tp + fp)
    ff.write('precision:' + str(precision) + '\n')

    mcc_0 = math.sqrt(abs((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) + 0.00001
    mcc = float(tp * tn - fp * fn) / mcc_0
    ff.write('mcc:' + str(mcc) + '\n')

    auc = metrics.roc_auc_score(aa_total, out_total)
    ff.write('AUC:' + str(auc) + '\n')

    ap = metrics.average_precision_score(aa_total, out_total)
    ff.write('AP:' + str(ap) + '\n')

    f1 = metrics.f1_score(aa_total, pred_total)
    ff.write('f1:' + str(f1) + '\n')

    out_total = out_total.tolist()
    aa_total = aa_total.tolist()
    with open(spath + str(fold) + '_result_no_surface.txt', 'w') as f:
        for i in range(len(out_total)):
            f.write(str(aa_total[i]) + '\t' + str(out_total[i]) + '\n')
