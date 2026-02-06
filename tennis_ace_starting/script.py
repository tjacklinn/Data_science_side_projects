import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load and investigate the data here:
data = pd.read_csv("/home/tjacklin/Data_Science_side_projects/tennis_ace_starting/tennis_stats.csv")
print(data.head())

# perform exploratory analysis here:
# check the column names
print(data.columns)

# check the data types
print(data.dtypes)

# Stats of each column
print(data.describe())

# Number of players
print("Number of players: ", data.Player.nunique())

# Number of years
print("Number of years: ", data.Year.nunique())

# Check if null values
print("Null values: ", data.isnull().sum().sum())

# Checking the max total winnings per player
total_wins_by_player = data.groupby("Player", as_index=False)["Winnings"].sum()
top20 = total_wins_by_player.sort_values("Winnings", ascending=False).head(20)

plt.bar(x=top20.Player, height=top20.Winnings)
plt.xticks(rotation=45, ha="right") 
plt.title("Total winnings per player")
plt.show()
plt.clf()

# Boxplots for serve stats for each year:
sns.boxplot(data=data, x="Year", y='FirstServe')
plt.xticks(rotation=45, ha="right") 
plt.show()
plt.clf()
sns.boxplot(data=data, x="Year", y='FirstServePointsWon')
plt.xticks(rotation=45, ha="right") 
plt.show()
plt.clf()
sns.boxplot(data=data, x="Year", y='SecondServePointsWon')
plt.xticks(rotation=45, ha="right") 
plt.show()
plt.clf()
sns.boxplot(data=data, x="Year", y='Aces')
plt.xticks(rotation=45, ha="right") 
plt.show()
plt.clf()
sns.boxplot(data=data, x="Year", y='DoubleFaults')
plt.xticks(rotation=45, ha="right") 
plt.show()
plt.clf()

# Computing and plotting correlation matrix
plt.figure(figsize=(20, 16))  
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of tennis stats")
plt.tight_layout()
plt.show()
plt.clf()

# There are many variables that have high correlation with "Wins/Losses/Winnings", such as Aces, BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ...




## perform single feature linear regressions here:
# Predicting the outcome based on BreakPointsOpportunities
features1 = data[['BreakPointsOpportunities']]
outcome1 = data[['Winnings']]

features1_train, features1_test, outcome1_train, outcome1_test = train_test_split(features1, outcome1, train_size = 0.8)

model1 = LinearRegression()
model1.fit(features1_train,outcome1_train)

model1.score(features1_test,outcome1_test)

prediction1 = model1.predict(features1_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome1_test,prediction1, alpha=0.4)
plt.title("Predicted vs real outcome based on BreakPointsOpportunities")
plt.show()
plt.clf()

# Predicting the outcome based on BreakPointsFaced
features2 = data[['BreakPointsFaced']]
outcome2 = data[['Winnings']]

features2_train, features2_test, outcome2_train, outcome2_test = train_test_split(features2, outcome2, train_size = 0.8)

model2 = LinearRegression()
model2.fit(features2_train,outcome2_train)

model2.score(features2_test,outcome2_test)

prediction2 = model2.predict(features2_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome2_test,prediction2, alpha=0.4)
plt.title("Predicted vs real outcome based on BreakPointsFaced")
plt.show()
plt.clf()

# Predicting the outcome based on ReturnGamesPlayed
features3 = data[['ReturnGamesPlayed']]
outcome3 = data[['Winnings']]

features3_train, features3_test, outcome3_train, outcome3_test = train_test_split(features3, outcome3, train_size = 0.8)

model3 = LinearRegression()
model3.fit(features3_train,outcome3_train)

model3.score(features3_test,outcome3_test)

prediction3 = model3.predict(features3_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome3_test,prediction3, alpha=0.4)
plt.title("Predicted vs real outcome based on ReturnGamesPlayed")
plt.show()
plt.clf()


def eval_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

metrics1 = eval_model(outcome1_test, prediction1)
metrics2 = eval_model(outcome2_test, prediction2)
metrics3 = eval_model(outcome3_test, prediction3)

print(metrics1)
print(metrics2)
print(metrics3)


# ReturnGamesPlayed has the highest R2, and lowest RMSE and MAE -> best of the tried models


## perform two feature linear regressions here:


# Combining two of the previous three features in different ways to find the best combination
features4 = data[['ReturnGamesPlayed',
'BreakPointsFaced']]
outcome4 = data[['Winnings']]

features4_train, features4_test, outcome4_train, outcome4_test = train_test_split(features4, outcome4, train_size = 0.8)

model4 = LinearRegression()
model4.fit(features4_train,outcome4_train)

model4.score(features4_test,outcome4_test)

prediction4 = model4.predict(features4_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome4_test,prediction4, alpha=0.4)
plt.title("Predicted vs real outcome based on ReturnGamesPlayed and BreakPointsFaced")
plt.show()
plt.clf()

features5 = data[['ReturnGamesPlayed',
'BreakPointsOpportunities']]
outcome5 = data[['Winnings']]

features5_train, features5_test, outcome5_train, outcome5_test = train_test_split(features5, outcome5, train_size = 0.8)

model5 = LinearRegression()
model5.fit(features5_train,outcome5_train)

model5.score(features5_test,outcome5_test)

prediction5 = model5.predict(features5_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome5_test,prediction5, alpha=0.4)
plt.title("Predicted vs real outcome based on ReturnGamesPlayed and BreakPointsOpportunities")
plt.show()
plt.clf()

features6 = data[['BreakPointsOpportunities',
'BreakPointsFaced']]
outcome6 = data[['Winnings']]

features6_train, features6_test, outcome6_train, outcome6_test = train_test_split(features6, outcome6, train_size = 0.8)

model6 = LinearRegression()
model6.fit(features6_train,outcome6_train)

model6.score(features6_test,outcome6_test)

prediction6 = model6.predict(features6_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome6_test,prediction6, alpha=0.4)
plt.title("Predicted vs real outcome based on BreakPointsOpportunities and BreakPointsFaced")
plt.show()
plt.clf()

metrics4 = eval_model(outcome4_test, prediction4)
metrics5 = eval_model(outcome5_test, prediction5)
metrics6 = eval_model(outcome6_test, prediction6)

print(metrics4)
print(metrics5)
print(metrics6)

# It is not clear which model is the best. The last one has the largest R2, but the other ones have smaller RMSE and MAE.


## perform multiple feature linear regressions here:
features7 = data[['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed']]
outcome7 = data[['Winnings']]

features7_train, features7_test, outcome7_train, outcome7_test = train_test_split(features7, outcome7, train_size = 0.8)

model7 = LinearRegression()
model7.fit(features7_train,outcome7_train)

model7.score(features7_test,outcome7_test)

prediction7 = model7.predict(features7_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome7_test,prediction7, alpha=0.4)
plt.title("Predicted vs real outcome based on 'Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed'")
plt.show()
plt.clf()

features8 = data[['Aces', 'BreakPointsOpportunities', 'DoubleFaults', 'ServiceGamesPlayed']]
outcome8 = data[['Winnings']]

features8_train, features8_test, outcome8_train, outcome8_test = train_test_split(features8, outcome8, train_size = 0.8)

model8 = LinearRegression()
model8.fit(features8_train,outcome8_train)

model8.score(features8_test,outcome8_test)

prediction8 = model8.predict(features8_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome8_test,prediction8, alpha=0.4)
plt.title("Predicted vs real outcome based on 'Aces', 'BreakPointsOpportunities', 'DoubleFaults', 'ServiceGamesPlayed'")
plt.show()
plt.clf()

features9 = data[['BreakPointsFaced', 'BreakPointsOpportunities', 'ReturnGamesPlayed', 'ServiceGamesPlayed']]
outcome9 = data[['Winnings']]

features9_train, features9_test, outcome9_train, outcome9_test = train_test_split(features9, outcome9, train_size = 0.8)

model9 = LinearRegression()
model9.fit(features9_train,outcome9_train)

model9.score(features9_test,outcome9_test)

prediction9 = model9.predict(features9_test)
plt.figure(figsize=(8, 6))
plt.scatter(outcome9_test,prediction9, alpha=0.4)
plt.title("Predicted vs real outcome based on 'BreakPointsFaced', 'BreakPointsOpportunities', 'ReturnGamesPlayed', 'ServiceGamesPlayed'")
plt.show()
plt.clf()

metrics7 = eval_model(outcome7_test, prediction7)
metrics8 = eval_model(outcome8_test, prediction8)
metrics9 = eval_model(outcome9_test, prediction9)

print(metrics7)
print(metrics8)
print(metrics9)

# Of these models, it is not clear which one is the best. The second one has the largest R2, but not the smallest RMSE or MAE.
