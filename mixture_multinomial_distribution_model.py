#####混合多項分布モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from scipy import stats
from numpy.random import *
from scipy import optimize
import seaborn as sns



####データの発生####
##データの設定
N = 1000   #サンプル数
k = 25   #変数数
seg = 5   #セグメント数
g = poisson(30, N)   #購買数


##セグメントの設定
seg_id = np.array([])
for i in range(seg):
 seg_id = np.append(seg_id, np.repeat(i, N/seg))


##頻度行列を発生
#パラメータの設定
p_runif = np.reshape(np.exp(normal(0.15, 0.8, seg*k)), (seg, k))
P_Seg = p_runif/np.reshape(np.repeat(np.sum(p_runif, axis=1), k), (seg, k))
pd.DataFrame(P_Seg)

#多項分布から頻度行列を発生
Y = np.zeros((N, k))
for i in range(N):
    index = int(seg_id[i])
    Y[i, :] = multinomial(g[i], P_Seg[index, :], 1)
pd.DataFrame(Y)


####EMアルゴリズムで混合多項分布モデルを推定####
##観測データの対数尤度と潜在変数zを計算するための関数
def LLobz(theta, Y, n, seg, r):

    #セグメント別のパラメータから尤度を計算
    LLind = np.zeros((Y.shape[0], seg))
    for j in range(seg):
        Li = stats.multinomial.pmf(Y, g, theta[j, :])
        LLind[:, j] = Li

    #潜在変数zと観測データの対数尤度を計算
    R = np.reshape(np.repeat(r, N), (N, seg), order='F')
    LLho =  R * LLind   #観測データの尤度
    z = LLho / np.reshape(np.repeat(np.sum(LLho, axis=1), seg), (N, seg))   #潜在変数zの計算
    LLosum = np.sum(np.log(np.sum(LLho, axis=1)))   #対数尤度を計算

    LL_list = [z, LLosum]   #リストに格納
    return(LL_list)


##初期値の設定
iter = 0
seg = 5   #セグメント数

#パラメータの初期設定
#確率パラメータの設定
p_runif = np.reshape(np.exp(normal(0.15, 0.9, seg*k)), (seg, k))
theta = p_runif / np.reshape(np.repeat(np.sum(p_runif, axis=1), k), (seg, k))
pd.DataFrame(theta)

#混合率の設定
r = np.array((0.25, 0.2, 0.1, 0.25, 0.2))


##潜在変数zと観測データの対数尤度の初期値を設定
L = LLobz(theta, Y, g, seg, r) 
z = L[0]   #潜在変数z 
LL1 = L[1]   #観測データの対数尤度

#更新ステータス
dl = 100   #EMステップでの対数尤度の差の初期値
tol = 0.01


##EMアルゴリズムで対数尤度を最大化
while abs(dl) >= tol:   #dlがtol以上の場合は繰り返す
    
    #Mステップの計算と最適化
    z = L[0]   #潜在変数zの出力
    for j in range(seg):
        Z_matrix = np.reshape(np.repeat(z[:, j], k), (N, k))
        theta[j, :] = np.sum(Z_matrix * Y, axis=0) / np.sum(z[:, j] * g)

    #混合率の推定
    r = np.sum(z, axis=0) / N

    #観測データの対数尤度を計算
    L = LLobz(theta, Y, g, seg, r)
    LL = L[1]
    iter = iter + 1
    dl = LL - LL1
    LL1 = LL
    print(np.round(LL, 2))


####推定結果と適合度####
##パラメータ推定値と真のパラメータの比較
theta_para = pd.DataFrame(np.round(np.concatenate((theta, P_Seg), axis=0), 3))   #確率パラメータの比較
z_para = pd.DataFrame(np.round(np.concatenate((z, np.reshape(seg_id, (N, 1))), axis=1), 3))   #潜在変数zと真のセグメントを比較
np.round(r, 3)   #混合率の推定結果

##適合度の計算
np.round(LL, 3)   #最大化された対数尤度
-2*LL + 2*(seg*theta.shape[1]+seg)   #AICの計算

