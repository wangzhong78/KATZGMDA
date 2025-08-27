import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

# 假设X为特征数据，y为标签数据
X = ...
y = ...

# 初始化逻辑斯蒂回归模型
model = LogisticRegression()

# 进行5次交叉验证，并获得预测概率
y_scores = cross_val_predict(model, X, y, cv=5, method="predict_proba")

# 计算每个fold的精确率、召回率和阈值
precisions, recalls, thresholds = precision_recall_curve(y, y_scores[:, 1])

# 将精确率、召回率和阈值保存成文件
np.save('precisions.npy', precisions)
np.save('recalls.npy', recalls)
np.save('thresholds.npy', thresholds)