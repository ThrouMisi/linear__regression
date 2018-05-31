#pandasを取り込む
import pandas as pd
#住宅価格に関するcsvファイルを取り込む
house = pd.read_csv('https://raw.githubusercontent.com/we-b/datasets_for_ai/master/cal_house.csv')
#説明変数と目的変数を定義する、
X = house[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = house['median_house_value']
print(X.shape)
print(y.shape)
print(X)
print(Y)
#データを訓練用とテスト用に分ける
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y ,test_size=0.3)
#線形回帰モデルを取り込み、データを分析させる
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
#訓練結果とテストの結果を表示
print(lr.score(X_train,y_train))
print(lr.score(X_test, y_test))
