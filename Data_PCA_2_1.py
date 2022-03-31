#Principal component analysis (PCA) 기초 (Data reduction)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#예제 데이터 생성
np.random.seed(5)
num_data = 500
w1, w2, w3 = 0.1, 0.2, 0.5
nn1, nn2, nn3 = 0.2, 0.1, 0.3
uu = np.random.rand(num_data) * np.pi
X = np.empty((num_data, 4))
X[:, 0] = np.cos(uu) + np.sin(uu)/2 + nn1 * np.random.randn(num_data)
X[:, 1] = np.sin(uu) + nn2 * np.random.randn(num_data)
X[:, 2] = np.sin(uu) + np.random.randn(num_data)*2
X[:, 3] = w1*X[:, 0] + w2*X[:, 1] + w3*X[:,2] + nn3 * np.random.randn(num_data)

#데이터 표준화 (Normalize, 평균 =0, 표준편차 1)
mean_X=np.mean(X,axis=0)
std_X=np.std(X,axis=0)
X_ = StandardScaler().fit_transform(X) #X_=(X-mean(X))/std(X)
# X_re = X_*std_X+mean_X (복원)

#Principal component 2 개 일 때 + normalized data 사용 했을 때
pca_result2 = PCA(n_components=2) #객체 생성
X_reduced2_   = pca_result2.fit_transform(X_) #차원 축소 실행
X_recovered2_ = pca_result2.inverse_transform(X_reduced2_) #차원 축소한 것을 다시 recover
X_recovered2  = X_recovered2_*std_X + mean_X #normalized 했던거 다시 recover

print("원본 데이터: \n", X)
print("PCA를 통해 차원 줄였다가 복원한 데이터: \n", X_recovered2)
print("PCA 결과 각 요소가 차지하는 variance 비율: \n",
      pca_result2.explained_variance_ratio_)