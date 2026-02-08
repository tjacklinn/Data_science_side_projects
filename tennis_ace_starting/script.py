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
def boxplot_by_year(data, col_name):
    sns.boxplot(data=data, x="Year", y=col_name)
    plt.xticks(rotation=45, ha="right") 
    plt.show()
    plt.clf()

# Boxplots for serve data
boxplot_by_year(data, 'FirstServe')
boxplot_by_year(data, 'FirstServePointsWon')
boxplot_by_year(data, 'SecondServePointsWon')
boxplot_by_year(data, 'Aces')
boxplot_by_year(data, 'DoubleFaults')

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
# functions for training and evaluation
def eval_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def train_and_evaluate(features, outcome, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, train_size=1 - test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'model': model,
        'scores': eval_model(y_test, preds),
        'predictions': preds,
        'true': y_test
    }

# Predicting the outcome based on BreakPointsOpportunities
result1 = train_and_evaluate(data[['BreakPointsOpportunities']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result1['true'], result1['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on BreakPointsOpportunities")
plt.show()
plt.clf()

# Predicting the outcome based on BreakPointsFaced
result2 = train_and_evaluate(data[['BreakPointsFaced']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result2['true'], result2['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on BreakPointsFaced")
plt.show()
plt.clf()

# Predicting the outcome based on ReturnGamesPlayed
result3 = train_and_evaluate(data[['ReturnGamesPlayed']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result3['true'], result3['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on ReturnGamesPlayed")
plt.show()
plt.clf()


metrics1 = eval_model(result1['true'], result1['predictions'])
metrics2 = eval_model(result2['true'], result2['predictions'])
metrics3 = eval_model(result3['true'], result3['predictions'])

print(metrics1)
print(metrics2)
print(metrics3)


# ReturnGamesPlayed has the highest R2, and lowest RMSE and MAE -> best of the tried models


## perform two feature linear regressions here:


# Combining two of the previous three features in different ways to find the best combination

result4 = train_and_evaluate(data[['ReturnGamesPlayed', 'BreakPointsFaced']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result4['true'], result4['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on ReturnGamesPlayed and BreakPointsFaced")
plt.show()
plt.clf()

result5 = train_and_evaluate(data[['ReturnGamesPlayed', 'BreakPointsOpportunities']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result5['true'], result5['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on ReturnGamesPlayed and BreakPointsOpportunities")
plt.show()
plt.clf()

result6 = train_and_evaluate(data[['BreakPointsOpportunities', 'BreakPointsFaced']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result6['true'], result6['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on BreakPointsOpportunities and BreakPointsFaced")
plt.show()
plt.clf()

metrics4 = eval_model(result4['true'], result4['predictions'])
metrics5 = eval_model(result5['true'], result5['predictions'])
metrics6 = eval_model(result6['true'], result6['predictions'])

print(metrics4)
print(metrics5)
print(metrics6)

# It is not clear which model is the best. The last one has the largest R2, but the other ones have smaller RMSE and MAE.


## perform multiple feature linear regressions here:
result7 = train_and_evaluate(data[['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result7['true'], result7['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on 'Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed'")
plt.show()
plt.clf()

result8 = train_and_evaluate(data[['Aces', 'BreakPointsOpportunities', 'DoubleFaults', 'ServiceGamesPlayed']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result8['true'], result8['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on 'Aces', 'BreakPointsOpportunities', 'DoubleFaults', 'ServiceGamesPlayed'")
plt.show()
plt.clf()

result9 = train_and_evaluate(data[['BreakPointsFaced', 'BreakPointsOpportunities', 'ReturnGamesPlayed', 'ServiceGamesPlayed']], data[['Winnings']])
plt.figure(figsize=(8, 6))
plt.scatter(result9['true'], result9['predictions'], alpha=0.4)
plt.title("Predicted vs real outcome based on 'BreakPointsFaced', 'BreakPointsOpportunities', 'ReturnGamesPlayed', 'ServiceGamesPlayed'")
plt.show()
plt.clf()

metrics7 = eval_model(result7['true'], result7['predictions'])
metrics8 = eval_model(result8['true'], result8['predictions'])
metrics9 = eval_model(result9['true'], result9['predictions'])

print(metrics7)
print(metrics8)
print(metrics9)

# Of these models, it is not clear which one is the best. The second one has the largest R2, but not the smallest RMSE or MAE.
