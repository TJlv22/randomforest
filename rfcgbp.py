import numpy as np
import pandas as pd
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna

data=pd.read_csv("gbpjpynow.csv",index_col='date', parse_dates=True)
data=data.drop("Unnamed: 5",axis=1)
data=data.sort_values("date")

#return
data["return"]=np.array(data["close"]-data["open"])
data["return%"]=np.array(data["close"]-data["open"])/data["open"]*100

#soar ddrop
data["soar%"]=(data["high"]-data["open"])/data["open"]*100
data["drop%"]=(data["low"]-data["open"])/data["open"]*100
#technical
close=np.array(data["close"])
#rsi 
data["rsi"]= ta.RSI(close, timeperiod=14)

#bollinger
upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=3, nbdevdn=3)
data["bb2"]= upper / close
data["bb-2"]= lower / close

#sma
data["sma5"]= ta.SMA(close, timeperiod=5) / close
data["sma20"]= ta.SMA(close, timeperiod=20) / close

#macd
data["macd"], _ , _= ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

#position
direction=np.array(data["return%"])
direction=np.where(direction<=0,-1,1)
data["direction"]=direction
data["future_direction"]=data["direction"].shift(-1)

data=data.dropna()

#train test split
x=data[["rsi","sma5","sma20","macd","bb2","bb-2","return%","soar%","drop%"]].values
y=data[["future_direction"]].values

xtrain,xtest,ytrain,ytest=train_test_split(x, y,train_size=0.9,shuffle=False)
ytrain=ytrain.reshape(len(ytrain),)
ytest=ytest.reshape(len(ytest),)
"""
model=RandomForestClassifier(random_state=1,min_samples_split=7, max_leaf_nodes=8, criterion="gini", n_estimators=200, max_depth=3,max_features=None)
model.fit(xtrain,ytrain)
result=model.predict(xtest)
print(accuracy_score(ytest,result))

"""
def objective(trial):
    min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 64, 4))
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 50,500,50))
    max_depth = trial.suggest_int("max_depth", 3,10)
    RFC= RandomForestClassifier(random_state=1, n_estimators = n_estimators, max_leaf_nodes = max_leaf_nodes, max_depth=max_depth, max_features=None,criterion=criterion,min_samples_split=min_samples_split)
    RFC.fit(xtrain, ytrain)
    result=RFC.predict(xtest)
    return 1.0 - accuracy_score(ytest,result)

study = optuna.create_study()
study.optimize(objective, n_trials = 100)
print(1-study.best_value)
print(study.best_params)












