#%%
# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수 test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다. (numpy의 array로 유지합니다.）

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

import xgboost as xgb

# 코드의 동작을 확인하기 위한 모델
class Model:
    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# 모델의 학습과 예측
# -----------------------------------
# 모델의 하이퍼 파라미터 지정
params = {'param1': 10, 'param2': 100}

# Model 클래스는 정의되어 있습니다.
# Model 클래스는 fit으로 학습하고, predict로 예측 결과의 확률을 출력합니다.

# 모델을 정의합니다.
model = Model(params)

# 학습 데이터에 대해 모델을 학습시킵니다.
model.fit(train_x, train_y)

# 테스트 데이터에 대해 예측결과를 출력합니다.
pred = model.predict(test_x)

# -----------------------------------
# 평가 및 검증
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 학습 데이터, 검증 데이터를 나누기 위해 인덱스를 작성.
# 학습 데이터를 4등분으로 분할하고, 그중 하나를 검증 데이터를 사용.
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 학습 데이터를 학습 데이터와 검증(평가)데이터로 나눈다.
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 모델을 정의합니다.
model = Model(params)

# 학습 데이터에 대해 모델을 학습 시킵니다.
# 모델에 따라 검증데이터를 함께 전달하여 점수를 모니터링이 가능합니다.
model.fit(tr_x, tr_y)

# 검증 데이터에 대해 예측해, 평가를 수행합니다.
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# -----------------------------------
# 교차 검증(CrossValidation)
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 학습 데이터를 4등분으로 나눠, 그중의 하나를 검증 데이터로 합니다.
# 어느 것을 검증 데이터로 할 것인가를 바꿔 학습 및 평가를 4회 실시합니다.
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# 교차 검증의 평균 스코어를 출력합니다.
print(f'logloss: {np.mean(scores):.4f}')
