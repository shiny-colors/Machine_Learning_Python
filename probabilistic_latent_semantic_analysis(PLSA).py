#####確率的潜在意味解析(トピックモデル)#####
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


####データを発生####
#データの設定
k = 8   #トピック数
d = 2000   #文書数
v = 300   #語彙数
w = poisson(250, d)   #1文書あたりの単語数
w_all = sum(w)   #総単語数


#ディレクリ分布のパラメータの設定
alpha0 = np.repeat(0.3, k)   #文書のディレリ事前分布のパラメータ
alpha1 = np.repeat(0.25, v)   #単語のディレクリ事前分布のパラメータ


#ディレクリ分布からパラメータを発生
theta = np.random.dirichlet(alpha0, d)   #文書のトピック分布をディレクリ乱数から発生
phi = np.random.dirichlet(alpha1, k)    #単語のトピック分布をデレクレリ乱数から発生
theta0 = theta
phi0 = phi



##多項分布から文書データを発生
WX = np.zeros((d, v), dtype='int')
Z = [i for i in range(d)]
vec = np.arange(0, k)

for i in range(d):
    #文書のトピックを生成
    z = multinomial(1, theta[i, :], w[i])   #文書のトピック分布を発生
    index_z = np.dot(z, vec)

    #トピック割当から単語を生成
    word = np.zeros((w[i], v))
    for j in range(w[i]):
        word[j, :] = multinomial(1, phi[index_z[j], :], 1)

    WX[i, :] = np.sum(word, axis=0)
    Z[i] = index_z



####トピックモデルのためのデータと関数の準備####
##それぞれの文書中の単語の出現をベクトルに並べる
##データ推定用IDを作成
ID_list = [i for i in range(d)]
wd_list = [i for i in range(d)]

#文書ごとに求人IDおよび単語IDを作成
for i in range(d):
    ID_list[i] = np.repeat(i, w[i])
    num1 = (WX[i, :] > 0) * np.arange(1, v+1)
    num2 = num1[num1!=0]
    W1 = WX[i, WX[i, :] > 0]
    number = np.repeat(num2, W1)
    wd_list[i] = number
    
#リストをベクトル変換
wd = np.zeros(sum(w), dtype='int')
ID_d = np.zeros(sum(w), dtype='int')
start = 0

for i in range(d):
    wd[start:start+w[i]] = wd_list[i] - 1
    ID_d[start:start+w[i]] = ID_list[i]
    start += w[i] 


##インデックスを作成
doc_list = [i for i in range(d)]
word_list = [i for i in range(v)]
index = np.array(range(w_all))

for i in range(d):
    doc_list[i] = index[ID_d==i]
for i in range(v):
    word_list[i] = index[wd==i]


##単語ごとに尤度と負担率を計算する関数
def burden_fr(w_all, theta, phi, wd, w, k):
    Bur = np.zeros((w_all, k))
    Bur0 = np.zeros((w_all, k))

    #負担係数を計算
    for j in range(k):
        Bi = np.repeat(theta[:, j], w) * phi[j, wd]
        Bur[:, j] = Bi

    #負担率の分母部分
    Bur_sums = np.sum(Bur, axis=1)
    for l in range(k):
        Bur0[:, l] = Bur_sums 

    #負担率と混合率を計算
    Br = Bur / Bur0   #負担率の計算
    r = np.sum(Br, axis=0) / np.sum(Br)   #混合率の計算
    return Br, Bur, r



####EMアルゴリズムの初期値を設定する####
##初期値をランダム化
#phiの初期値
phi = np.zeros((k, v))
freq_v = np.zeros((k, v))
rand_v = np.zeros((k, v))
freq = np.sum(WX, axis=0)   #単語の出現数

for i in range(k):
    rand_v[i, :] = abs(freq + np.array(np.round(normal(loc=0, scale=np.mean(freq)/2, size=v)), dtype='int'))   #ランダム化
    phi[i, :] = rand_v[i, :] / np.sum(rand_v[i, :])   #phiを初期化

#thetaの初期値
alpha = np.repeat(0.3, k)
theta = np.random.dirichlet(alpha, d)   #文書のトピック分布の初期値をディレクリ乱数から発生



##パラメータの更新
#負担率と尤度の更新
bfr = burden_fr(w_all, theta, phi, wd, w, k)
Br = bfr[0]
r = bfr[2]


#thetaの更新
tsum = np.zeros((d, k))
W = np.zeros((d, k))
for j in range(k):
    W[:, j] = w

for i in range(d):
    tsum[i, :] = np.sum(Br[doc_list[i], :], axis=0)
theta = tsum / W   #thetaの更新


#phiの更新
vf = np.zeros((v, k))
for i in range(v):
    vf[i, :] = np.sum(Br[word_list[i], :], axis=0)

V0 = np.sum(vf, axis=0)
V = np.zeros((k, v))
for j in range(v):
    V[:, j] = V0
phi = vf.T / V


#対数尤度の計算
LLS = np.sum(np.log(np.sum(bfr[1], axis=1)))
print(LLS)


####EMアルゴリズムでパラメータを更新####
#更新ステータス
iter = 1
dl = 100   #EMステップでの対数尤度の差の初期値
tol = 0.5
LLo = LLS   #対数尤度の初期値
LLw = LLS
abs(dl)


##パラメータを更新
while abs(dl) >= tol:

    #負担率と尤度の更新
    bfr = burden_fr(w_all, theta, phi, wd, w, k)
    Br = bfr[0]
    r = bfr[2]

    #thetaの更新
    tsum = np.zeros((d, k))
    W = np.zeros((d, k))
    for j in range(k):
        W[:, j] = w

    for i in range(d):
        tsum[i, :] = np.sum(Br[doc_list[i], :], axis=0)
    theta = tsum / W   #thetaの更新

    #phiの更新
    vf = np.zeros((v, k))
    for i in range(v):
        vf[i, :] = np.sum(Br[word_list[i], :], axis=0)

    V0 = np.sum(vf, axis=0)
    V = np.zeros((k, v))
    for j in range(v):
        V[:, j] = V0

    phi = vf.T / V   #phiの更新

    #対数尤度の更新
    LLS = np.sum(np.log(np.sum(bfr[1], axis=1)))

    iter <- iter+1
    dl = LLS-LLo
    LLo = LLS
    LLw <- np.array((LLw, LLo))
    print(LLo)

####推定結果と統計量####
np.round(pd.DataFrame(np.concatenate((phi.T, phi0.T), axis=1)), 3)
np.round(pd.DataFrame(np.concatenate((theta, theta0), axis=1)), 3)

