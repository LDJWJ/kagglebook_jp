import numpy as np
import pandas as pd

# -----------------------------------
# 와이드 포맷(Wide Format), 롱 포맷(Long Format)
# -----------------------------------

# 와이드 포맷의 데이터를 읽기
df_wide = pd.read_csv('../input/ch03/time_series_wide.csv', index_col=0)
# 인덱스의 형태를 날짜형으로 변경
df_wide.index = pd.to_datetime(df_wide.index)

print(df_wide.iloc[:5, :3])
'''
              A     B     C
date
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
2016-07-04  937  5451  1977
2016-07-05  881  4729  1975
'''

# 롱 포맷으로 변환
df_long = df_wide.stack().reset_index(1)
df_long.columns = ['id', 'value']

print(df_long.head(10))
'''
           id  value
date
2016-07-01  A    532
2016-07-01  B   3314
2016-07-01  C   1136
2016-07-02  A    798
2016-07-02  B   2461
2016-07-02  C   1188
2016-07-03  A    823
2016-07-03  B   3522
2016-07-03  C   1711
2016-07-04  A    937
...
'''

# 와이드 포맷으로 되돌리다.
df_wide = df_long.pivot(index=None, columns='id', values='value')

#%%
# -----------------------------------
# lag 변수
# -----------------------------------
# 와이드 포맷(Wide Format)의 데이터을 준비
x = df_wide
# -----------------------------------
# x는 와이드 포맷의 데이터 프레임
# 인덱스가 날짜 등의 시간, 열이 사용자나 점포 등이고 값이 매출 등에 주목하는 변수를 나타내는 것으로 한다. 
# 1기 이전의 lag를 획득 
x_lag1 = x.shift(1)

# 7기 이전의 lag를 획득
x_lag7 = x.shift(7)

#%%
# -----------------------------------
# 1기 이전전부터 3기 기간의 이동평균을 산출
x_avg3 = x.shift(1).rolling(window=3).mean()

# -----------------------------------
# 1기 이전부터 7기 기간의 최대치를 산출 
x_max7 = x.shift(1).rolling(window=7).max()

# -----------------------------------
# 7기 이전, 14기 이전, 21기 이전, 28기 이전의 수치 평균
x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0

#%%
# -----------------------------------
# 1기 앞쪽의 값을 취득
x_lead1 = x.shift(-1)

#%%
# -----------------------------------
# lag 변수
# -----------------------------------
# 데이터 읽어오기
train_x = pd.read_csv('../input/ch03/time_series_train.csv')
event_history = pd.read_csv('../input/ch03/time_series_events.csv')
train_x['date'] = pd.to_datetime(train_x['date'])
event_history['date'] = pd.to_datetime(event_history['date'])
# -----------------------------------

# train_x는 학습 데이터로, 사용자 ID, 날짜를 열로 갖는 DataFrame로 한다. 
# event_history는 과거에 개최한 이벤트의 정보로 날짜, 이벤트를 열로 가진 DataFrame로 한다.

# occurrences는 날짜, 세일 개최 여부를 열로 가진 DataFrame이 된다.
dates = np.sort(train_x['date'].unique())
occurrences = pd.DataFrame(dates, columns=['date'])
sale_history = event_history[event_history['event'] == 'sale']
occurrences['sale'] = occurrences['date'].isin(sale_history['date'])

# 누적합을 얻으므로 각각의 날짜로의 누적 출현횟수를 나타내도록 한다.
# occurrences는 날짜, 세일 누적 출현 횟수를 열로 갖는 데이터 프레임(DataFrame)이 된다.
occurrences['sale'] = occurrences['sale'].cumsum()

# 날짜를 키로서 학습 데이터와 결합한다. 
train_x = train_x.merge(occurrences, on='date', how='left')
