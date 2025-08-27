import numpy as np
import torch as t
import numpy as np
import scipy.sparse as sp


def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = t.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = t.where(t.isinf(D_5), t.full_like(D_5, 0), D_5)
    L_D_11 = t.mm(D_5, L_D_1)
    L_D_11 = t.mm(L_D_11, D_5)
    return L_D_11


def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = t.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne


def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl

def getMinKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)  # 请确保有相应的normalized_embedding函数定义
    dist_matrix = pairwise_distance(y, y)  # 计算样本之间的距离矩阵
    kernel_matrix = t.exp(-gamma * dist_matrix)  # 使用RBF核函数计算核矩阵
    return kernel_matrix

def pairwise_distance(x, y):
    x_norm = t.sum(x ** 2, dim=-1, keepdim=True)
    y_t = y.permute(1, 0)
    y_norm = t.sum(y ** 2, dim=-1, keepdim=True)
    dist = x_norm - 2 * t.matmul(x, y_t) + y_norm.t()
    return dist

def katzhmda_py( Wdr, beta):
    #cc = np.matmul(Wdr, Wdr.T)
    cc = (Wdr @ Wdr.T).detach().numpy()
    #matDT = cc
    #cc=t.mm(Wdr,Wdr.T)
    #drug_sim = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\drugsimilarity.txt', delimiter='\t')
    #mic_sim = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\microbesimilarity.txt', delimiter='\t')

    #matDT = interaction = np.loadtxt("C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\drug_microbe_matrix2.txt", delimiter='\t')
    matDT = interaction = np.loadtxt("C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\drug_mic_matrix8.txt",delimiter='\t')

    drug_sim =cc[:1373,:1373].copy()
    mic_sim =cc[-173:,-173:].copy()

    # matDT = cc
    # 假设GIPSim函数执行的是某种矩阵操作，可进行类似的替代
    #KD, KM = GIPSim(matDT, 1, 1)
    # 将numpy数组转换为PyTorch张量
    #matDT_tensor = t.from_numpy(matDT)
    #matDT_array = matDT.detach().numpy()
    #matDT_tensor = t.from_numpy(matDT_array)
    # 确保KD和KM也是PyTorch张量
    KD = t.tensor(drug_sim)
    KM = t.tensor(mic_sim)
    #matDT_array = matDT.detach().numpy()
    matDT_array = matDT
    matDT = t.from_numpy(matDT_array)
    print(matDT.shape)
    # 进行矩阵乘法运算
    #result = t.matmul(KM, matDT.T) + t.matmul(matDT.T, KD)
    result = t.matmul(KM.double(), matDT.T.double()) + t.matmul(matDT.T.double(), KD.double())
    Sk2 = beta * matDT.T + beta ** 2 * result
    # 进行矩阵乘法运算
    #Sk2 = beta * matDT.T + beta** 2 * (KM.dot(matDT.T) + matDT.T.dot(KD))
    #Sk2 = beta * matDT_tensor.T + beta ** 2 * (KM.dot(matDT_tensor.T) + matDT_tensor.T.dot(KD))

    #Sk2 =beta * matDT.T + beta**2 * (KM.dot(matDT.T) + matDT.T.dot(KD))
    recMatrix = Sk2.T
    T = np.vstack([np.hstack([KM, recMatrix.T]), np.hstack([recMatrix, KD])])
    return T

def katzhmda_py1( Wdr, beta,drug_mic_matrix):
    #cc = np.matmul(Wdr, Wdr.T)
    cc = (Wdr @ Wdr.T).detach().numpy()


    drug_len = drug_mic_matrix.shape[0]
    dis_len = drug_mic_matrix.shape[1]
    matDT=drug_mic_matrix
    drug_sim = cc[:drug_len, :drug_len].copy()
    mic_sim = cc[-dis_len:, -dis_len:].copy()


    #cc = (Wdr @ Wdr.T).detach().numpy()
    #matDT = cc
    #cc=t.mm(Wdr,Wdr.T)
    #drug_sim = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\drugsimilarity.txt', delimiter='\t')
    #mic_sim = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\microbesimilarity.txt', delimiter='\t')

    #matDT = interaction = np.loadtxt("C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\MDAD\drug_microbe_matrix2.txt", delimiter='\t')

    #drug_sim =cc[:1373,:1373].copy()
    #mic_sim =cc[-173:,-173:].copy()

    # matDT = cc
    # 假设GIPSim函数执行的是某种矩阵操作，可进行类似的替代
    #KD, KM = GIPSim(matDT, 1, 1)
    # 将numpy数组转换为PyTorch张量
    #matDT_tensor = t.from_numpy(matDT)
    #matDT_array = matDT.detach().numpy()
    #matDT_tensor = t.from_numpy(matDT_array)
    # 确保KD和KM也是PyTorch张量
    KD = t.tensor(drug_sim)
    KM = t.tensor(mic_sim)
    #matDT_array = matDT.detach().numpy()
    matDT_array = matDT
    matDT = t.from_numpy(matDT_array)

    # 进行矩阵乘法运算
    #result = t.matmul(KM, matDT.T) + t.matmul(matDT.T, KD)
    result = t.matmul(KM.double(), matDT.T.double()) + t.matmul(matDT.T.double(), KD.double())
    Sk2 = beta * matDT.T + beta ** 2 * result
    # 进行矩阵乘法运算
    #Sk2 = beta * matDT.T + beta** 2 * (KM.dot(matDT.T) + matDT.T.dot(KD))
    #Sk2 = beta * matDT_tensor.T + beta ** 2 * (KM.dot(matDT_tensor.T) + matDT_tensor.T.dot(KD))

    #Sk2 =beta * matDT.T + beta**2 * (KM.dot(matDT.T) + matDT.T.dot(KD))
    recMatrix = Sk2.T
    T = np.vstack([np.hstack([KM, recMatrix.T]), np.hstack([recMatrix, KD])])
    return T


def GIPSim(interaction, gamadd, gamall):
    # interaction: 疾病和miRNA之间的关系矩阵，列为miRNA，行为疾病
    def GIP_Calculate(M):  # 计算高斯核相似性
        l = np.size(M, axis=1)
        sm = []
        m = np.zeros((l, l))
        for i in range(l):
            tmp = (np.linalg.norm(M[:, i])) ** 2
            sm.append(tmp)
        gama = l / np.sum(sm)
        for i in range(l):
            for j in range(l):
                m[i, j] = np.exp(-gama * ((np.linalg.norm(M[:, i] - M[:, j])) ** 2))
        return m


    nd, nl = interaction.shape

    # 计算用于高斯核计算的gamad
    sd = np.zeros(nd)
    for i in range(nd):
        sd[i] = np.linalg.norm(interaction[i, :]) ** 2
    gamad = nd / np.sum(sd) * gamadd*sd[i]

    # 计算用于高斯核计算的gamal
    sl = np.zeros(nl)
    for i in range(nl):
        sl[i] = np.linalg.norm(interaction[:, i]) ** 2
    gamal = nl / np.sum(sl) * gamall*sl[i]

    # 计算疾病之间的相似性的高斯核：kd
    kd = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            kd[i, j] = np.exp(-gamad * (np.linalg.norm(interaction[i, :] - interaction[j, :]) ** 2))

    # 计算miRNA之间的相似性的高斯核：kl
    kl = np.zeros((nl, nl))
    for i in range(nl):
        for j in range(nl):
            kl[i, j] = np.exp(-gamal * (np.linalg.norm(interaction[:, i] - interaction[:, j]) ** 2))

    return kd, kl

def RWR(SM):
    alpha = 0.5
    E = np.identity(len(SM))  # 单位矩阵
    M = np.zeros((len(SM), len(SM)))
    s=[]
    for i in range(len(M)):
        for j in range(len(M)):
            M[i][j] = SM[i][j] / (np.sum(SM[i, :]))
    for i in range(len(M)):
        e_i = E[i, :]
        p_i1 = np.copy(e_i)
        for j in range(10):
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s.append(p_i1)
    return s




def kernelToDistance(k):
    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    return t.DoubleTensor([t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])


def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[t.nonzero(k, as_tuple=False)[0]]
    K[t.where(K == 0)] = min_v
    D = t.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S


class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.drug_size = drug_size
        self.mic_size = mic_size
        self.F1 = 128
        self.F2 = 64
        self.F3 = 32
        self.k_fold = 5
        self.epoch = 10
        self.learn_rate = 0.01
        self.seed =5
        self.h1_gamma = 2 ** (-5)
        self.h2_gamma = 2 ** (-5)
        self.h3_gamma = 2 ** (-5)

        self.lambda1 = 2 ** (-2)
        self.lambda2 = 2 ** (-4)
        self.beta=0.1




