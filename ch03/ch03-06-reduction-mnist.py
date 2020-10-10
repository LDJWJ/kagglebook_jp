# ---------------------------------
# 데이터 등의 사전 준비
# ----------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MNIST 데이터 가시화

# keras.datasets를 이용하여 MNIST 데이터를 다운로드 실시 
from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 2차원 데이터로 변경 
train_x = train_x.reshape(train_x.shape[0], -1)

# 상위 1000건으로 축소시키기
train_x = pd.DataFrame(train_x[:1000, :])
train_y = train_y[:1000]

#%%
# -----------------------------------
# PCA
# -----------------------------------
from sklearn.decomposition import PCA

# 학습 데이터를 기반으로 한 PCA에 의한 변환 정의 
pca = PCA()
x_pca = pca.fit_transform(train_x)

# 분류 후의 데이터로 2차원으로 그리기
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_pca[mask, 0], x_pca[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

#%%
# -----------------------------------
# LDA (Linear Discriminant Analysis)
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 클래스를 가장 잘 나누는 두 축을 선형 판별 분석으로 도출
lda = LDA(n_components=2)
x_lda = lda.fit_transform(train_x, train_y)

# 학급별로 분류하여 이차원 데이터를 도출
# 잘 분할하고 있으나 목적변수를 이용하고 있기 대문에 다른 것과 비교하여 매우 유리한 조건임을 주의
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_lda[mask, 0], x_lda[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

#%%
# -----------------------------------
# t-sne
# -----------------------------------
from sklearn.manifold import TSNE

# t-sne에 의한 변환
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(train_x)

# 클래스마다 나눠, 2차원 그래프 그리기 
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_tsne[mask, 0], x_tsne[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

#%%
# -----------------------------------
# UMAP
# -----------------------------------
import umap

# UMAP에 의한 변환
um = umap.UMAP()
x_umap = um.fit_transform(train_x)

# 클래스마다 나눠, 2차원 그래프 그리기 
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_umap[mask, 0], x_umap[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()
