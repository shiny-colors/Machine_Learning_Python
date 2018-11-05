#####混合多変量正規分布モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from scipy.special import gammaln
from scipy.misc import factorial
from pandas.tools.plotting import scatter_matrix
from numpy.random import *
from scipy import optimize
import seaborn as sns


####任意の相関行列(分散共分散行列)を作成する関数####
##任意の相関行列を作る関数
def CorM(col, lower, upper, eigen_lower, eigen_upper):
    #相関行列の初期値を定義する
    cov_vec = (upper - lower) *rand(col*col) + lower   #相関係数の乱数ベクトルを作成
    rho = np.reshape(np.array(cov_vec), (col, col)) * np.tri(col)   #乱数ベクトルを下三角行列化
    Sigma = np.diag(np.diag(rho + rho.T) + 1) - (rho + rho.T)   #対角成分を1にする
    
    #相関行列を正定値行列に変更
    #固有値分解を実行
    eigen = scipy.linalg.eigh(Sigma)
    eigen_val = eigen[0] 
    eigen_vec = eigen[1]
    
    #固有値が負の数値を正にする
    for i in range(eigen_val.shape[0]-1):
        if eigen_val[i] < 0:
            eigen_val[i] = (eigen_upper - eigen_lower) * rand(1) + eigen_lower
            
    #新しい相関行列の定義と対角成分を1にする
    Sigma = np.dot(np.dot(eigen_vec, np.diag(eigen_val)), eigen_vec.T)
    normalization_factor = np.dot(pow(np.diag(Sigma), 0.5)[:, np.newaxis], pow(np.diag(Sigma), 0.5)[np.newaxis, :])
    Cor = Sigma / normalization_factor
    return Cor

##相関行列から分散共分散行列に変換する関数
def covmatrix(Cor, sigma_lower, sigma_upper):
    sigma = (sigma_upper - sigma_lower) * rand(np.diag(Cor).shape[0]) + sigma_lower
    sigma_factor = np.dot(sigma[:, np.newaxis], sigma[np.newaxis, :])
    Cov = Cor * sigma_factor
    return Cov


####データの発生####
##データの設定
n = 1000   #セグメントのサンプル数
seg = 4   #セグメント数
N = n*seg   #総サンプル数
k = 5   #パラメータ数


##セグメント割当の設定
seg_id = np.array([])
for i in range(seg):
    seg_id = np.append(seg_id, np.repeat(i+1, n))


##多変量正規分布からセグメントごとにデータを発生させる
#パラメータの設定
#パラメータと応答変数の格納用配列
Cor0 = np.zeros((k, k, seg))
Cov0 = np.zeros((k, k, seg))
Mu0 = np.zeros((seg, k))
Y = np.zeros((N, k))

#セグメントごとにパラメータを設定して応答変数を発生させる
for i in range(seg):
    lower = uniform(-0.6, 0.25)
    upper = uniform(0.25, 0.75)
    Cor0[:, :, i] = CorM(col=k, lower=-0.55, upper=0.8, eigen_lower=0.01, eigen_upper=0.2)
    Cov0[:, :, i] = covmatrix(Cor=Cor0[:, :, i], sigma_lower=0.7, sigma_upper=1.75)
    Mu0[i, :] = uniform(-4, 4, k)
    Y[seg_id==i+1, :] = np.random.multivariate_normal(Mu0[i, :], Cov0[:, :, i], n)


##発生させた変数の集計
#散布図行列をプロット
Y_pd = pd.DataFrame(np.concatenate((seg_id[:, np.newaxis], Y), axis=1))
scatter_matrix(Y_pd[[1, 2, 3, 4, 5]], diagonal='kde', color='k', alpha=0.3)
plt.show()


####EMアルゴリズムで混合多変量正規分布を推定####
##多変量正規分布の尤度関数を定義
def dmv(x, mu, Cov, k):
    er = x - mu
    Cov_inv = np.linalg.inv(Cov) 
    LLo = 1 / (np.sqrt(pow((2 * np.pi), k) * np.linalg.det(Cov))) * np.exp(np.dot(np.dot(-er, Cov_inv), er) / 2)
    return(LLo)


##観測データの対数尤度と潜在変数zの定義
def LLobz(Mu, Cov, r, Y, N, seg):
    LLind = np.zeros((N, seg))
    alpha = pow(10, -305)

    #多変量正規分布の尤度をセグメントごとに計算
    for s in range(seg):
        mean_vec = Mu[s, :]
        cov = Cov[:, :, s]

        for i in range(N):
            LLind[i, s] = dmv(Y[i, :], mean_vec, cov, k) + alpha   #尤度が桁落ちしないように微小な尤度を加える

    #対数尤度と潜在変数zの計算
    LLho = np.reshape(np.repeat(r, N), (N, seg), order='F') * LLind
    z = LLho / np.reshape(np.repeat(np.sum(LLho, axis=1), seg), (N, seg))   #潜在変数z
    LL = np.sum(np.log(np.sum(np.reshape(np.repeat(r, N), (N, seg), order='F') * LLind, axis=1)))
    LL_list = [z, LL]   #リストに格納
    return(LL_list)


####EMアルゴリズムで混合多変量正規分布を推定する####
##更新ステータス
dl = 100   #EMステップでの対数尤度の差の初期化
tol = 1 
iter = 1
max_iter = 100


##パラメータの初期値を設定
#全サンプルでのパラメータ
mu_all = np.mean(Y, axis=0)
var_all = np.cov(Y, rowvar=0, bias=1)

#パラメータの初期値の設定
Cov = np.zeros((k, k, seg))
Mu = np.zeros((seg, k))

#セグメントごとにパラメータの初期値を設定
for i in range(seg):
    Cov[:, :, i] = var_all
    Mu[i, :] = mu_all + uniform(-0.5, 0.5, k)

#混合率の初期値
r = np.array((0.3, 0.3, 0.2, 0.2))

#対数尤度と潜在変数zの初期化
L = LLobz(Mu, Cov, r, Y, N, seg)
LL1 = L[1]
z = L[0]


##EMアルゴリズムによる推定
while abs(dl) >= tol:   #dlがtol以上の場合は繰り返す

    #Mステップの計算
    z = L[0]   #潜在変数zの出力

    #多変量正規分布のパラメータを推定
    for i in range(seg):
        #平均ベクトルを推定
        zm = np.reshape(np.repeat(z[:, i], Y.shape[1]), (N, Y.shape[1])) 
        Mu[i, :] = np.sum(zm * Y, axis=0) / (Y.shape[0] * r[i])

        #分散共分散行列を推定
        mu_vec = np.reshape(np.repeat(Mu[i, :], N), (N, Y.shape[1]), order="F")
        z_er = zm * Y - zm * mu_vec 
        Cov[:, :, i] = np.dot(z_er.T, z_er) / np.sum(z[:, i])

    #混合率の推定
    r = np.sum(z, axis=0) / Y.shape[0]

    #Eステップの計算
    L = LLobz(Mu, Cov, r, Y, N, seg)
    LL = L[1]
    iter = iter + 1
    dl = LL - LL1
    LL1 = LL
    print(np.round(LL, 2))


####推定されたパラメータの確認と適合度####
##推定されたパラメータ
Mu_par = pd.DataFrame(np.round(np.concatenate((Mu, Mu0)), 3))   #平均ベクトルの推定値
Cov_par = np.round(np.concatenate((Cov,Cov0), axis=0), 3)   #分散共分散行列の推定値
r_par = np.round(r, 3)   #混合率の推定値
z_par = pd.DataFrame(np.round(z, 3))   #潜在変数zの推定値

##適合度の計算
np.round(LL, 3)   #最大化された対数尤度
-2*LL + 2*(seg*Mu.shape[1]+seg*sum(range(k+1)))   #AICの計算
-2*LL + np.log(Y.shape[0])*(seg*Mu.shape[1]+seg*sum(range(k+1)))   #BICの計算

