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

# neural net용 데이터
train_nn = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x_nn = train_nn.drop(['target'], axis=1)
train_y_nn = train_nn['target']
test_x_nn = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

#%%
# ---------------------------------
# hold-out 데이터로의 예측값을 사용한 앙상블
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_index = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_index]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_index]
tr_x_nn, va_x_nn = train_x_nn.iloc[tr_idx], train_x_nn.iloc[va_index]

# models.py에 Model1_1, Model1_2, Model2을 정의하고 있는 것으로 한다.
# 각 클래스는 fit으로 학습하고, fit으로 학습한 후, predict로 예측값 확률을 출력한다.
from models import Model1Xgb, Model1NN, Model2Linear

# 첫번째 층의 모델
# 학습 데이터로 학습하고, hold-out 데이터와 테스트 데이터에 대한 예측값을 출력
model_1a = Model1Xgb()
model_1a.fit(tr_x, tr_y, va_x, va_y)
va_pred_1a = model_1a.predict(va_x)
test_pred_1a = model_1a.predict(test_x)

model_1b = Model1NN()
model_1b.fit(tr_x_nn, tr_y, va_x_nn, va_y)
va_pred_1b = model_1b.predict(va_x_nn)
test_pred_1b = model_1b.predict(test_x_nn)

# hold-out 데이터에서의 정밀도를 평가
print(f'logloss: {log_loss(va_y, va_pred_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(va_y, va_pred_1b, eps=1e-7):.4f}')
# logloss : 0.3009
# logloss : 0.2785

# hold-out 데이터와 테스트 데이터에 대한 예측값을 특징(feature)로 데이터 프레임을 생성.
va_x_2 = pd.DataFrame({'pred_1a': va_pred_1a, 'pred_1b': va_pred_1b})
test_x_2 = pd.DataFrame({'pred_1a': test_pred_1a, 'pred_1b': test_pred_1b})

# 두번째 층의 모델
# Hold-out 데이터 모두에서 학습하고 있으므로 평가할 수 없다.
# 평가를 하기 위해서는 Hold-out 데이터를 추가적으로 교차검증하는 방법을 고려할 수 있다.
model2 = Model2Linear()
model2.fit(va_x_2, va_y, None, None)
pred_test_2 = model2.predict(test_x_2)
