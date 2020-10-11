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

# 학습 데이터를 학습 데이터와 검증(평가)데이터로 나눈다.
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 특징(feature)의 목록에 대하여 정밀도를 평가하는 evaluate 함수의 정의
import xgboost as xgb
from sklearn.metrics import log_loss


def evaluate(features):
    dtrain = xgb.DMatrix(tr_x[features], label=tr_y)
    dvalid = xgb.DMatrix(va_x[features], label=va_y)
    params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
    num_round = 10  # 実際にはもっと多いround数が必要
    early_stopping_rounds = 3
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_round,
                      evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=0)
    va_pred = model.predict(dvalid)
    score = log_loss(va_y, va_pred)

    return score

#%%
# ---------------------------------
# Greedy Forward Selection
# ----------------------------------

best_score = 9999.0
selected = set([])

print('start greedy forward selection')

while True:

    if len(selected) == len(train_x.columns):
        # 모든 특징(feature)이 선정되어 종료
        break

    scores = []
    for feature in train_x.columns:
        if feature not in selected:
            # 특징(feature)의 리스트에 대해서 정도를 평가하는 evaluate 함수가 있는 것으로 한다.
            fs = list(selected) + [feature]
            score = evaluate(fs)
            scores.append((feature, score))

    # 스코어는 낮은 쪽이 좋다고 가정
    b_feature, b_score = sorted(scores, key=lambda tpl: tpl[1])[0]
    if b_score < best_score:
        selected.add(b_feature)
        best_score = b_score
        print(f'selected:{b_feature}')
        print(f'score:{b_score}')
    else:
        # 어떤 특징(feature)을 추가해도 스코어가 오르지 않으므로 종료
        break

print(f'selected features: {selected}')
# selected features: 
# {'sex', 'medical_keyword_2', 'age', 'weight', 'height', 'product', 
# 'medical_info_a2', 'medical_keyword_4', 'medical_info_a1', 'medical_keyword_3', 'medical_keyword_5'}

#%%
# ---------------------------------
# Greedy Forward Selection의 단순화 기법
# ----------------------------------

best_score = 9999.0
candidates = np.random.RandomState(71).permutation(train_x.columns)
selected = set([])

print('start simple selection')
for feature in candidates:
    # 특징(feature)의 리스트에 대해서 정밀도를 평가하는 evaluate함수로 수행
    fs = list(selected) + [feature]
    score = evaluate(fs)

    # 스코어는 낮은 쪽이 좋다고 가정한다.
    if score < best_score:
        selected.add(feature)
        best_score = score
        print(f'selected:{feature}')
        print(f'score:{score}')

print(f'selected features: {selected}')
# selected features: 
# {'sex', 'medical_keyword_2', 'medical_info_b1', 'age', 'medical_keyword_5', 
# 'weight', 'height', 'medical_keyword_8', 'product', 'medical_info_c1', 'medical_info_a2', 'medical_keyword_1', 'medical_keyword_3', 'medical_info_a1', 'yearmonth', 'medical_keyword_6', 'medical_keyword_4'}
