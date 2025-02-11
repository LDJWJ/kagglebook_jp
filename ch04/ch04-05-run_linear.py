# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수 test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다. (numpy의 array로 유지합니다.）

# one-hot encoding 된 것을 불러오기

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# 학습 데이터을 학습 데이터와 검증 데이터로 나눈다.
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#%%
# -----------------------------------
# 선형 모델의 구현
# -----------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# 데이터의 스케일링
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x)
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# 선형 모델의 구축 및 학습
model = LogisticRegression(C=1.0)
model.fit(tr_x, tr_y)

# 검증 데이터의 스코어 확인
# predict_proba을 사용해 확률을 출력할 수 있습니다. (predict에서는 두 가지 값의 예측치가 출력됩니다.)
va_pred = model.predict_proba(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')
# logloss: 0.3719

# 예측
pred = model.predict(test_x)
