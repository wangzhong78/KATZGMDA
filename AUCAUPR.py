import numpy as np
import random

from GAT import *
from utils import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, \
    precision_recall_curve



pre_matrix=np.loadtxt("C:\新建文件夹\MKGCN-main\code\pre_matrix.txt")
fixmatrix=np.loadtxt("C:\新建文件夹\MNNMDA-main\model_zoo\matlab_models\MNNMDA\MatPredict_DRMNN-HMDAD.npy.txt")

scores=[]
labels=[]

#score=torch.sigmoid(torch.FloatTensor(np.dot(Fr, Fm.T)))
score=torch.sigmoid(torch.FloatTensor(pre_matrix+fixmatrix))
np.savetxt('C:\TYQ-master (2)\data\MDAD\sorce.txt', score)
score=score.numpy()
for i in range(1373):
    for j in range(173):
        scores.append(score[i][j])
        labels.append(A[i][j])
fpr, tpr, threshold = roc_curve(labels, scores)
AUPR = average_precision_score(labels,scores)
auc_val = auc(fpr, tpr)
print(auc_val,AUPR)