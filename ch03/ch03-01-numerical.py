#%%
# ---------------------------------
# 데이터 등 기본 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series을 사용합니다. (numpy의 array로 사용하기도 합니다.）

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# 설명용으로 학습 데이터와 테스트 데이터를 원본 상태를 저장해 둡니다.
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# 학습 데이터와 테스트 데이터를 반환하는 함수
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 변환할 수치형 변수 리스트를 저장
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

#%%
# -----------------------------------
# 표준화
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 학습 데이터를 기반으로 복수 열의 표준화를 수행
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

#%%
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 학습 데이터와 테스트 데이터를 결합하여 복수열의 표준화를 정의
scaler = StandardScaler()
scaler.fit(pd.concat([train_x[num_cols], test_x[num_cols]]))

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

#%%
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 학습 데이터와 테스트 데이터를 각각 표준화 수행(나쁜 예)
scaler_train = StandardScaler()
scaler_train.fit(train_x[num_cols])
train_x[num_cols] = scaler_train.transform(train_x[num_cols])

scaler_test = StandardScaler()
scaler_test.fit(test_x[num_cols])
test_x[num_cols] = scaler_test.transform(test_x[num_cols])

#%%
# -----------------------------------
# Min-Max 스케일링
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import MinMaxScaler

# 학습 데이터를 기반으로 복수열의 Min-Max 스케일링 정의
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

#%%
# -----------------------------------
# 로그(대수) 변환
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# 단순히 값에 로그를 취함
x1 = np.log(x)

# 1을 더한 뒤에 로그를 취함
x2 = np.log1p(x)

# 절대값의 로그를 취한 후, 원래 값의 부호를 추가
x3 = np.sign(x) * np.log(np.abs(x))

#%%
# -----------------------------------
# Box-Cox 변환
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------

# 양의 정수 값만을 취하는 변수를 변환 대상으로 목록에 저장
# 또한, 결측치의 값을 포함하는 경우는, (~(train_x[c] <= 0.0)).all() 등으로 할 필요가 있으므로 주의??
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

# 학습 데이터를 기반으로 복수열의 Box-Cox 변환 정의
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

#%%
# -----------------------------------
# Yeo-Johnson 변환
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------

from sklearn.preprocessing import PowerTransformer

# 학습 데이터를 기반으로 여러 열의 Yeo-Johnson 변환 정의
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

#%%
# -----------------------------------
# clipping
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
# 열마다 학습 데이터 1%, 99%의 지점을 확인
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# 1％점 이하의 점은 1%점으로, 99%점 이상의 값은 99%점으로 clipping한다.
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

#%%
# -----------------------------------
# binning
# -----------------------------------
x = [1, 7, 5, 4, 6, 3]

# pandas의 cut함수로 binning을 수행

# bin의 수를 지정할 경우,
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 변환된 값은 세개의 bin 중에 어느것이 들어가는지 나타낸다.

# bin의 범위를 지정할 경우(3.0 이하, 3.0보다 크고 5.0보다 이하, 5.0보다 큼)
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 변환된 값은 세 bin중 어느 것에 들어갔는지를 나타낸다.

#%%
# -----------------------------------
# 순위 변환
# -----------------------------------
x = [10, 20, 30, 0, 40, 40]

# pandas의 rank함수에서 순위로 변환하다.
rank = pd.Series(x).rank()
print(rank.values)

# 시작이 1, 동 순위가 있을 경우에는 평균 순위가 된다.
# [2. 3. 4. 1. 5.5 5.5]

# numpy의 argsort 함수를 2회 적용하는 방법으로 순위를 변환한다.
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# 시작이 0, 동순위가 있을 경우는 어느 쪽이든 상위가 된다.
# [1 2 3 0 4 5]

#%%
# -----------------------------------
# RankGauss
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import QuantileTransformer

# 학습 데이터를 기반으로 복수열의 RankGauss를 통한 변환 정의
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 변환 후 데이터로 각 열을 치환
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
