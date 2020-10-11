#%%
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# -----------------------------------
# 학습 데이터(train.csv), 테스트 데이터(test.csv) 읽기
# -----------------------------------
# 학습 데이터, 테스트 데이터 읽기
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# 학습 데이터를 특징(feature)과 목적 변수로 나누기
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 테스트 데이터는 독립변수만 있기 때문에, 그대로 사용.
test_x = test.copy()

#%%
# -----------------------------------
# 특징(feature, 피처) 만들기
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 변수 PassengerId을 제거
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 변수 [ Name, Ticket, Cabin ]을 제거
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 범주형 변수에 label encoding 을 적용하여 수치로 변환
for c in ['Sex', 'Embarked']:
    # 학습 데이터를 기반으로 어떻게 변환 할지를 최적화 시킨다.
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # 학습 데이터, 테스트 데이터를 변환
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))


# -----------------------------------
# 모델 검증
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold


# 각 fold의 평가 점수(스코어)를 위한 빈 리스트 생성
scores_accuracy = []
scores_logloss = []

# 교차 검증(Cross-validation)을 수행
# 01 학습 데이터를 4개로 분할
# 02 그 중의 하나를 평가용 데이터 셋으로 한다.
# 03 이후 평가용 데이터의 조각을 하나씩 옆으로 옮겨가며 검증을 수행
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # # 학습 데이터를 학습 데이터와 평가용 데이터 셋으로 나눈다.
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 모델 학습을 수행
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # 평가용 데이터의 예측 결과를 확률로 출력
    va_pred = model.predict_proba(va_x)[:, 1]

    # 평가용 데이터의 점수(스코어)를 계산
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 각 fold의 점수(스코어)를 저장
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

#각 fold의 점수 평균을 출력.
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148（이 책의 수치와 다를 가능성이 있습니다.）


# -----------------------------------
# 모델 튜닝
# -----------------------------------
import itertools

# 튜닝을 하기 위한 후보 파라미터를 준비
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 하이퍼 파라미터 값의 조합
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 각 파라미터의 조합(params)과 그에 대한 스코어를 보존(scores)하는 빈 리스트 
params = []
scores = []

# 각 파라미터 조합별로 교차 검증(Cross-validation) 평가 수행
for max_depth, min_child_weight in param_combinations:

    score_folds = []

    # 교차 검증(Cross-validation)을 수행
    # 학습 데이터를 4개로 분할한 후, 
    # 그중 하나를 평가용 데이터로 한 후, 평가을 수행하고, 이를 데이터를 바꾸어 가면서 반복. 
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 학습 데이터를 학습 데이터와 검증용 데이터로 분할
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 모델 학습 수행
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # 검증용 데이터의 스코어를 계산한 후, 저장
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # 각 fold의 스코어의 평균을 구한다.
    score_mean = np.mean(score_folds)

    # 파라미터를 조합, 그에 대한 스코어를 저장한다.
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 가장 스코어가 좋은 것을 베스트 파라미터로 한다.
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0의 스코어가 가장 좋음.

