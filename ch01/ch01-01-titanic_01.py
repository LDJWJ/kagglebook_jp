#%%
import numpy as np
import pandas as pd

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
# 모델 만들기
# -----------------------------------
from xgboost import XGBClassifier

# 모델 생성 및 학습 데이터를 이용한 모델 학습
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# 테스트 데이터의 예측 결과를 확률로 출력한다.
pred = model.predict_proba(test_x)[:, 1]

# 테스트 데이터의 예측 결과를 두개의 값(1,0)으로 변환
pred_label = np.where(pred > 0.5, 1, 0)

# 제출용 파일의 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# 스코어 ：0.77751（이 책에서 수치와 다를 가능성이 있습니다.）