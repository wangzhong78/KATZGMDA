import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
plt.switch_backend('Tkagg')

# 模型A的AURoc曲线数据
#model_A_fpr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#model_A_tpr = [0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.95, 1.0]
# 从Excel文件中读取数据
#data = np.loadtxt('C:\新建文件夹\MKGCN-main\code\\fpr.xlsx', delimiter='\t', skiprows=1, encoding='utf-8')

# 获取第一行数据作为X轴
#x_data = data[0]
model_A_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\KATZHMDA_fpr.txt')
# 获取第二行数据作为Y轴
#y_data = data[1]
model_A_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\KATZHMDA_tpr.txt')
# 绘制图表
auc_val_A=auc(model_A_fpr,model_A_tpr)




# 模型B的AURoc曲线数据
model_B_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\LRLSHMDA_fpr.txt')
# 获取第二行数据作为Y轴
#y_data = data[1]
model_B_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\LRLSHMDA_tpr.txt')
# 绘制图表
auc_val_B=auc(model_B_fpr,model_B_tpr)


model_C_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MNNMDA_fpr.txt')
# 获取第二行数据作为Y轴
#y_data = data[1]
model_C_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MNNMDA_tpr.txt')
# 绘制图表
auc_val_C=auc(model_C_fpr,model_C_tpr)

model_D_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\NTSHMDAfpr2.txt')
# 获取第二行数据作为Y轴
#y_data = data[1]
model_D_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\NTSHMDAtpr2.txt')
# 绘制图表
auc_val_D=auc(model_D_fpr,model_D_tpr)

model_E_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\GSAMDA_fpr.txt')
# 获取第二行数据作为Y轴
#y_data = data[1]
model_E_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\GSZMDA_tpr.txt')
# 绘制图表
auc_val_E=auc(model_E_fpr,model_E_tpr)


model_F_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MKGCNfpr_values.txt')

model_F_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MKGCNtpr_values.txt')

auc_val_F=auc(model_F_fpr,model_F_tpr)


model_G_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\Lagcnx_ROC.txt')

model_G_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\Lagcny_ROC.txt')

auc_val_G=auc(model_G_fpr,model_G_tpr)





# 预测的模型AURoc曲线数据
model_H_fpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\fpr8.txt')
# 获取第二行数据作为Y轴
#y_data = data[1]
model_H_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\tpr8.txt')
# 绘制图表
auc_val_H=auc(model_H_fpr,model_H_tpr)







# 绘制AURoc曲线
plt.plot(model_A_fpr, model_A_tpr, color="r", lw=1, label='KATZHMDA(Auc=%0.4f)' % auc_val_A)
plt.plot(model_B_fpr, model_B_tpr, color="g", lw=1, label='LRLSHMDA(Auc=%0.4f)' % auc_val_B)
plt.plot(model_C_fpr, model_C_tpr, color="y", lw=1, label='MNNMDA(Auc=%0.4f)' % auc_val_C)
#plt.plot(model_D_fpr, model_D_tpr, color="m", lw=1, label='NTSHMDA(Auc=%0.4f)' % auc_val_D)
plt.plot(model_E_fpr, model_E_tpr, color="cyan", lw=1, label='GSAMDA(Auc=%0.4f)' % auc_val_E)
#plt.plot(model_F_fpr, model_F_tpr, color="deeppink", lw=1, label='MKGCN(Auc=%0.4f)' % auc_val_F)
plt.plot(model_G_fpr, model_G_tpr, color="powderblue", lw=1, label='LAGCN(Auc=%0.4f)' % auc_val_G)

plt.plot(model_H_fpr, model_H_tpr, color="b", lw=1, label='KATZGMDA(Auc=%0.4f)' % auc_val_H)
#plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.legend(loc="lower right")
# 设置图形标签
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AURoc Curve Comparison')
plt.legend()
# 显示图形
plt.show()







model_A_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\KATZHMDA_tpr.txt')
model_A_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\KATZHMDA_Precision.txt')
aupr_A = auc(model_A_tpr, model_A_precision)

model_B_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\LRLSHMDAtpr2.txt')
model_B_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\LRLSHMDApre2.txt')
aupr_B = auc(model_B_tpr, model_B_precision)

model_C_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MNNMDA_tpr.txt')
model_C_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MNNMDA_precision.txt')
aupr_C = auc(model_C_tpr, model_C_precision)

model_D_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\NTSHMDAtpr2.txt')
model_D_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\NTSHMD_precision.txt')
aupr_D = auc(model_D_tpr, model_D_precision)

model_E_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\GSA_recall1.txt')
model_E_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\GSA_precision1.txt')
aupr_E = auc(model_E_tpr, model_E_precision)

model_F_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MKGCNtpr_values.txt')
model_F_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\MKGCNprecision_list.txt')
aupr_F = auc(model_F_tpr, model_F_precision)

model_G_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\Lagcnx_PR.txt')
model_G_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\Lagcny_PR.txt')
aupr_G = auc(model_G_tpr, model_G_precision)






model_H_tpr=np.loadtxt('C:\新建文件夹\MKGCN-main\code\\recall_list8.txt')
model_H_precision=np.loadtxt('C:\新建文件夹\MKGCN-main\code\precision_list8.txt')
aupr_H = auc(model_H_tpr, model_H_precision)

# 画AUPR曲线
plt.figure()
plt.plot(model_A_tpr, model_A_precision, color='r', lw=1, label='KATZHMDA (Aupr = %0.4f)' % aupr_A)
plt.plot(model_B_tpr, model_B_precision, color='g', lw=1, label='LRLSHMDA (Aupr = %0.4f)' % aupr_B)
plt.plot(model_C_tpr, model_C_precision, color='y', lw=1, label='MNNMDA (Aupr = %0.4f)' % aupr_C)
#plt.plot(model_D_tpr, model_D_precision, color='m', lw=1, label='NTSHMD (Aupr = %0.4f)' % aupr_D)
plt.plot(model_E_tpr, model_E_precision, color='m', lw=1, label='GSAMDA (Aupr = %0.4f)' % aupr_E)


#plt.plot(model_F_tpr, model_F_precision, color='deeppink', lw=1, label='MKGCN (Aupr = %0.4f)' % aupr_F)
plt.plot(model_G_tpr, model_G_precision, color='powderblue', lw=1, label='LAGCN (Aupr = %0.4f)' % aupr_G)



plt.plot(model_H_tpr, model_H_precision, color='b', lw=1, label='KATZGMDA(Aupr = %0.4f)' % aupr_H)
#plt.fill_between(model_A_tpr,model_A_precision, alpha=0.2, color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc="lower left")
plt.show()