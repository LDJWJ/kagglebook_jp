# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수 test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다. (numpy의 array로 유지합니다.）

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')

# 학습 데이터를 학습 데이터와 검증(평가)데이터로 나눈다.
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# tensorflow의 경고 메시지 제어
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%%
# -----------------------------------
# 신경망 파라미터의 튜닝의 예
# -----------------------------------
from hyperopt import hp
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler

# 기본이 되는 파라미터 
base_param = {
    'input_dropout': 0.0,
    'hidden_layers': 3,
    'hidden_units': 96,
    'hidden_activation': 'relu',
    'hidden_dropout': 0.2,
    'batch_norm': 'before_act',
    'optimizer': {'type': 'adam', 'lr': 0.001},
    'batch_size': 64,
}

# 탐색할 파라미터 공간을 지정
param_space = {
    'input_dropout': hp.quniform('input_dropout', 0, 0.2, 0.05),
    'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),
    'hidden_units': hp.quniform('hidden_units', 32, 256, 32),
    'hidden_activation': hp.choice('hidden_activation', ['prelu', 'relu']),
    'hidden_dropout': hp.quniform('hidden_dropout', 0, 0.3, 0.05),
    'batch_norm': hp.choice('batch_norm', ['before_act', 'no']),
    'optimizer': hp.choice('optimizer',
                           [{'type': 'adam',
                             'lr': hp.loguniform('adam_lr', np.log(0.00001), np.log(0.01))},
                            {'type': 'sgd',
                             'lr': hp.loguniform('sgd_lr', np.log(0.00001), np.log(0.01))}]),
    'batch_size': hp.quniform('batch_size', 32, 128, 32),
}


class MLP:

    def __init__(self, params):
        self.params = params
        self.scaler = None
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):

        # 파라미터
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])

        # 표준화
        self.scaler = StandardScaler()
        tr_x = self.scaler.fit_transform(tr_x)
        va_x = self.scaler.transform(va_x)

        self.model = Sequential()

        # 입력층
        self.model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1],)))

        # 중간층(은닉층)
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))

        # 출력층
        self.model.add(Dense(1, activation='sigmoid'))

        # 최적화(옵티마이저)
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 목적함수, 평가지표등의 설정
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])

		# 에폭수, 얼리스토핑ポック数、アーリーストッピング
		# 너무 epoch을 크게하면 작은 학습률일 때 끝나지 않을 수 있기 때문에 주의	
        nb_epoch = 200
        patience = 20
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        # 학습의 실행
        history = self.model.fit(tr_x, tr_y,
                                 epochs=nb_epoch,
                                 batch_size=batch_size, verbose=1,
                                 validation_data=(va_x, va_y),
                                 callbacks=[early_stopping])

    def predict(self, x):
        # 예측	
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred

#%%
# -----------------------------------
# 파라미터 튜닝 실행

from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import log_loss


def score(params):
	# 파라미터 세트를 지정했을 때, 최소화해야 할 함수를 지정
	# 모델의 파라미터 탐색에서는 모델에게 파라미터를 지정하여 학습・예측하게 한 경우의 스코어로 한다.
    model = MLP(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')

    # 정보를 기록하기
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}

# hyperopt에 의한 파라미터 탐색의 실행
max_evals = 10
trials = Trials()
history = []
fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 기록한 정보에서 파라미터와 스코어를 출력한다.
# trials에서도 정보를 취득할 수 있지만 파라미터를 취득하기 어렵다.
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')
# val_acc: 0.8228
# params: {'batch_norm': 'no', 'batch_size': 64.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.05, 'hidden_layers': 3.0, 'hidden_units': 128.0, 'input_dropout': 0.2, 'optimizer': {'lr': 4.246552647978557e-05, 'type': 'sgd'}}, logloss: 0.4176
# best params:{'batch_norm': 'no', 'batch_size': 64.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.25, 'hidden_layers': 2.0, 'hidden_units': 192.0, 'input_dropout': 0.15000000000000002, 'optimizer': {'lr': 0.008124231674520142, 'type': 'adam'}}, score:0.2757