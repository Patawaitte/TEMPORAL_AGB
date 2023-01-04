
import os, sys
import subprocess as sp
import glob
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

df = gpd.read_file("C:/Lidar/shp/PEP_ABG_MEL_2949_intersept_withname.shp")
df['idjoin']=(df['filenam']).str[-18:]

mymetrics = pd.read_csv('C:/Lidar/samplemetrics_firstreturn_zone9.csv')  
print(mymetrics)
mymetrics["idjoin"]=(mymetrics['DataFile']).str[-18:]
#metrics = metrics.replace(0, np.nan, inplace=True)
print(mymetrics)
mymetrics.replace('-nan(ind)', np.nan, inplace=True)


#metrics.drop(['Int CV', 'Int skewness', 'Int kurtosis', 'Int L CV', 'Int L CV', 'Int L kurtosis'], axis=1, inplace=True)
mymetrics.drop(mymetrics.columns[mymetrics.apply(lambda col: col.isnull().sum() > 0)], axis=1, inplace=True)
mymetrics = mymetrics.loc[:, (mymetrics != 0).any(axis=0)]

join_metric = pd.merge(df[["id_pe_mes", "BM_vivant_", "Placette_5", "date_sond","idjoin" ]], mymetrics, on="idjoin", how='left')
print(list(join_metric.columns.values))
#join_metric.to_csv('C:/Lidar/samplemetrics_join.csv')  
#join_metric=join_metric.loc[join_metric['Placette_5'] == 'R']
#join_metric=join_metric[['id_pe_mes', 'BM_vivant_', 'Placette_5', 'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle', 'Elev P80','Percentage first returns above 2.00']]

join_metric=join_metric[['id_pe_mes', 'BM_vivant_', 'Placette_5', 'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle', 'Total return count','Elev maximum', 'Elev MAD median','Elev L kurtosis', 'Elev P01', 'Elev P05', 'Elev P50', 'Elev P90','Elev CURT mean CUBE', 'Elev SQRT mean SQ', 'Percentage first returns above 2.00', 'Percentage all returns above 2.00' , 'First returns above mean']]
#print(list(join_metric.columns.values))

feature_names = list(join_metric.iloc[:, 8:109].columns.values)

X = join_metric.iloc[:, 8:109].values
y = join_metric.iloc[:, 1].values

print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


regressor = RandomForestRegressor(n_estimators=500, random_state=0, max_depth=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))


print((cross_val_score(regressor, X_train, y_train, cv=10, scoring = 'r2')))
print((np.mean(cross_val_score(regressor, X_train, y_train, cv=10, scoring = 'r2'))))

plt.scatter(y_test, y_pred)
plt.axline([0, 0], slope=1)
plt.show()
result = permutation_importance(
    regressor, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
forest_importances = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

