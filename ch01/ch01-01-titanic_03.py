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

#%%
# -----------------------------------
# 로지스틱 회귀용 특징(feature) 작성
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 원래 데이터를 복사한다.
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 변수PassengerId을 제외한다.
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 변수Name, Ticket, Cabin을 제외한다.
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# one-hot encoding을 수행
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# one-hot encoding의 더미 변수의 열명을 작성한다.
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# one-hot encoding에 의한 변수를 수행
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# one-hot encoding의 수행이 끝난 변수를 제외한다.
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# one-hot encoding으로 변환된 변수를 결합한다.
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# 수치 변수의 결손 값을 학습 데이터의 평균으로 채우기
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 변수Fare을 로그(대수) 변환
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# 앙상블(ensemble)
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboost 모델
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# 로지스틱 회귀 모델
# xgboost 모델과는 다른 특성(feature)를 넣을 필요에 따라 train_x2, test_x2를 생성.
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 예측 결과의 가중 평균을 취하다.
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)

# 옮긴이 추가
# 제출용 파일의 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_second.csv', index=False)
# score : 0.78468