import numpy as np
import pandas as pd

# ---------------------------------
# 랜덤 포레스트(RandomForest) 특징(feature)의 중요도
# ---------------------------------
# train_x는 학습 데이터, train_y는 목적 변수
# 결측치가 처리할 수 없기 때문에, 결측치값을 보완한 데이터를 읽어들인다.
train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']

#%%
# ---------------------------------
from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트(RandomForest)
clf = RandomForestClassifier(n_estimators=10, random_state=71)
clf.fit(train_x, train_y)
fi = clf.feature_importances_

# 중요도를 상위에 출력
idx = np.argsort(fi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], fi[idx][:5]
print('random forest importance')
print(top_cols, top_importances)

# ['medical_info_a1' 'weight' 'age' 'medical_info_a2' 'height'] 
# [0.12604874 0.11164059 0.07741062 0.07132529 0.05367491]

#%%
# ---------------------------------
# xgboost의 특징(feature)의 중요도
# ---------------------------------
# train_x는 학습 데이터, train_y는 목적변수 
train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
# ---------------------------------
import xgboost as xgb

# xgboost
dtrain = xgb.DMatrix(train_x, label=train_y)
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50
model = xgb.train(params, dtrain, num_round)

# 중요도의 상위를 출력합니다.
fscore = model.get_score(importance_type='total_gain')
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print('xgboost importance')
print(fscore[:5])

# xgboost importance
# [('weight', 2614.0292876953), ('medical_info_a1', 2240.9029884895026), 
# ('height', 1973.3420545093588), ('age', 1442.832677605481), ('medical_info_a2', 1150.6861460469188)]
