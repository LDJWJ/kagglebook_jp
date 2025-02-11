import numpy as np
import pandas as pd

# -----------------------------------
# 데이터 결합
# -----------------------------------
# 데이터 읽어오기
train = pd.read_csv('../input/ch03/multi_table_train.csv')
product_master = pd.read_csv('../input/ch03/multi_table_product.csv')
user_log = pd.read_csv('../input/ch03/multi_table_log.csv')

# 그림 형식의 데이터 프레임이 있다고 합니다.
# train : 학습 데이터（사용자ID, 상품ID, 목적변수 등의 열이 있음）
# product_master: 상품 마스터（상품 ID와 상품의 정보를 나타내는 줄이 있음.）
# user_log : 사용자 행동의 로그 데이터（사용자 ID와 각 행동의 정보를 나타내는 열이 있음）

# 상품 마스터를 학습 데이터와 결합
train = train.merge(product_master, on='product_id', how='left')

# 로그 데이터의 사용자별 행의 수를 집계해, 학습 데이터와 결합
user_log_agg = user_log.groupby('user_id').size().reset_index().rename(columns={0: 'user_count'})
train = train.merge(user_log_agg, on='user_id', how='left')
