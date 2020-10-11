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

# 학습 데이터을 학습 데이터와 검증 데이터로 나눈다.
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#%%
# -----------------------------------
# lightgbm 모델 활용
# -----------------------------------
import lightgbm as lgb
from sklearn.metrics import log_loss

# 특징(feature)과 목적변수를 lightgbm의 데이터 구조로 변환
lgb_train = lgb.Dataset(tr_x, tr_y)
lgb_eval = lgb.Dataset(va_x, va_y)

# 하이퍼 파라미터 설정
params = {'objective': 'binary', 'seed': 71, 'verbose': 0, 'metrics': 'binary_logloss'}
num_round = 100

# 학습 실행
# 카테고리 변수를 파라미터로 지정
# 검증 데이터도 모델에게 건네주어 학습 진행과 함께 점수가 어떻게 달라지는지 모니터링
categorical_features = ['product', 'medical_info_b2', 'medical_info_b3']
model = lgb.train(params, lgb_train, num_boost_round=num_round,
                  categorical_feature=categorical_features,
                  valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval])

# 검증 데이터에서의 스코어 확인
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')
# logloss : 0.2225

# 예측
pred = model.predict(test_x)
