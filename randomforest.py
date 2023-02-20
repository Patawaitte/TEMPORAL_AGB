
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
import seaborn as sns
from sklearn.model_selection import cross_val_score
from shapely.geometry import Point

df = gpd.read_file("C:/Lidar/shp/Placette_synchrone_1an_lazname_AGB_UA_clean.shp")
df['idjoin']= df['idmes']
#df['idjoin']= pd.to_numeric(df['idmes'],errors='coerce').astype(int)

#df.idjoin = df.idmes.apply(int)
#df=(df['idmes']).astype(str).str.split('.', expand=True)
#df = df.drop(df[df.bd_placett.isnull()].index)

#dfua = df[(df["TERRITOIRE"] == '03351') | (df["TERRITOIRE"] == '03171')| (df["TERRITOIRE"] == '03153')]


#PEP = gpd.read_file("C:/Lidar/shp/Biomass_Placette-20230104T163943Z-001/Biomass_Placette/PEP_vivant.shp")
#PET4 = gpd.read_file("C:/Lidar/shp/Biomass_Placette-20230104T163943Z-001/Biomass_Placette/PET4_vivant.shp")
#PET5 = gpd.read_file("C:/Lidar/shp/Biomass_Placette-20230104T163943Z-001/Biomass_Placette/PET5_vivant.shp")
#print(PET4.columns.values)

#PEP['id_pe_mes']= PEP['id_pe_mes'].astype(float)
#PET4['id_pe']= PET4['id_pe'].astype(float)
#PET5['id_pe']= PET5['id_pe'].astype(float)

#PEP.dropna(subset=['bio_comp_1'], inplace=True)
#PET4.dropna(subset=['bio_comp_1'], inplace=True)
#PET5.dropna(subset=['bio_comp_1'], inplace=True)

#MERGE PEP, PET4 et PET5
#merge1= pd.merge(df[["idmes", "biomass_ha", "date_sond","idjoin", "type_couv", "geometry", "st_x", "st_y", "TERRITOIRE"]], PET4, left_on="idmes",right_on="id_pe", how='left', suffixes=('', '_PET4'))
#merge2= pd.merge(merge1, PET5, left_on="idmes",right_on="id_pe", how='left', suffixes=('','_PET5'))
#merge3= pd.merge(merge2, PEP, left_on="idmes",right_on="id_pe_mes", how='left', suffixes=('','_PEP'))#.rename(columns={"date_sond_x": "date_sond", "type_couv_x": "type_couv"})

#merge3.to_file("C:/Lidar/merge3.shp", driver='ESRI Shapefile')

#merge3['biomass_ha'] = np.where(merge3["bio_comp_1"].notnull(), merge3['bio_comp_1'], merge3['biomass_ha'])
#merge3['biomass_ha'] = np.where(merge3["bio_comp_1_PET5"].notnull(), merge3['bio_comp_1_PET5'], merge3['biomass_ha'])
#merge3['biomass_ha'] = np.where(merge3["bio_comp_1_PEP"].notnull(), merge3['bio_comp_1_PEP'], merge3['biomass_ha'])



#MERGE ONLY PEP
#merge3= pd.merge(df, PEP, left_on="idmes",right_on="id_pe_mes", how='left').rename(columns={"date_sond_x": "date_sond", "type_couv_x": "type_couv"})
#merge3['biomass_ha'] = np.where(merge3["bio_comp_1"].notnull(), merge3['bio_comp_1'], merge3['biomass_ha'])
#merge3 = merge3.drop(merge3[merge3.bio_comp_1.isnull()].index)
#merge3 = merge3.drop(merge3[merge3.type_couv.isnull()].index)

#print(merge3.columns.values)
#print(df['idjoin'])
#print(merge3['biomass_ha'])


#delete error data
df['AGB_viv']= df['AGB_viv'].astype(float)
df = df.drop(df[df.AGB_viv.isnull()].index)

df = df.drop(df[df.AGB_viv >1000].index)
df=df.loc[df['Type'] == 'PEP']
df = df.drop(df[df.type_couv.isnull()].index)

#merge3['geometry'] = merge3.apply(lambda x: Point((float(x.st_x), float(x.st_y))), axis=1)
#gdf = gpd.GeoDataFrame(merge3, geometry='geometry')

#merge3.to_file("C:/Lidar/AGB_allmerge.shp", driver='ESRI Shapefile')
#plt.hist(merge3['biomass_ha'],bins=50)
#plt.show()
mymetrics = pd.read_csv('C:/Lidar/samplemetrics_allreturn.csv')  
print(mymetrics)
mymetrics.replace('-nan(ind)', np.nan, inplace=True)
#mymetrics['FileTitle'] = mymetrics['FileTitle'].astype("string")
mymetrics['idjoin']=(mymetrics['FileTitle']).str.slice(5, 20).astype(float)
print(mymetrics['idjoin'])


#metrics.drop(['Int CV', 'Int skewness', 'Int kurtosis', 'Int L CV', 'Int L CV', 'Int L kurtosis'], axis=1, inplace=True)
mymetrics.drop(mymetrics.columns[mymetrics.apply(lambda col: col.isnull().sum() > 0)], axis=1, inplace=True)
mymetrics = mymetrics.loc[:, (mymetrics != 0).any(axis=0)]

join_metric = pd.merge(df[["idmes", "AGB_viv", "date_sond","idjoin", "type_couv", ]], mymetrics, on="idjoin", how='left')
print(list(join_metric.columns.values))
join_metric= join_metric.dropna(axis=0)

#join_metric.to_csv('C:/Lidar/samplemetrics_join.csv')  
#join_metric=join_metric.loc[join_metric['type_couv'] == 'R']
#join_metric=join_metric[['id_pe_mes', 'BM_vivant_', 'Placette_5', 'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle', 'Elev P80','Percentage first returns above 2.00']]
#join_metric=join_metric[['idmes', 'AGB_viv',  'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle',  'Elev MAD median','Elev L2','Elev L skewness', 'Canopy relief ratio']]

#join_metric=join_metric[['idmes', 'biomass_ha',  'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle','Elev variance', 'Elev MAD median','Elev MAD median','Elev L2','Elev L4','Canopy relief ratio', 'Percentage all returns above mean' ]]
join_metric=join_metric[['idmes', 'AGB_viv',  'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle', 'Elev MAD median',  'Elev L skewness' , 'First returns above mean']]

#join_metric=join_metric[['idmes', 'AGB_viv',  'date_sond', 'idjoin', 'Identifier', 'DataFile', 'FileTitle', 'Total return count','Elev maximum', 'Elev MAD median','Elev L kurtosis', 'Elev P01', 'Elev P05', 'Elev P50', 'Elev P90','Elev CURT mean CUBE', 'Elev SQRT mean SQ', 'Percentage first returns above 2.00', 'Percentage all returns above 2.00' , 'First returns above mean']]
#print(list(join_metric.columns.values))

feature_names = list(join_metric.iloc[:, 7:109].columns.values)

test=join_metric[['AGB_viv','Elev MAD median',  'Elev L skewness' , 'First returns above mean']]





X = join_metric.iloc[:, 7:109].values
#y = join_metric.iloc[:, 1].values
y=join_metric[['idmes','AGB_viv']]




g= sns.pairplot(test,hue = 'AGB_viv', diag_kind= 'hist',
             vars=test[: 1:4],
             plot_kws=dict(alpha=0.5),
             diag_kws=dict(alpha=0.5))
plt.show()


print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


regressor = RandomForestRegressor(n_estimators=500, random_state=0, max_depth=5)
regressor.fit(X_train, y_train['AGB_viv'])
y_pred = regressor.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns = ['pred'])
result= pd.concat([y_test.reset_index(), y_pred_df], axis=1)

result['diff']= result['AGB_viv']-result['pred']
result.to_csv('C:/Lidar/result_test.csv')


print('Mean Absolute Error:', metrics.mean_absolute_error(result['AGB_viv'], result['pred']))
print('Mean Squared Error:', metrics.mean_squared_error(result['AGB_viv'], result['pred']))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(result['AGB_viv'], result['pred'])))
print('R2:', metrics.r2_score(result['AGB_viv'], result['pred']))


#print((cross_val_score(regressor, X_train, y_train['AGB_viv'], cv=10, scoring = 'r2')))
#print((np.mean(cross_val_score(regressor, X_train, y_train['AGB_viv'], cv=10, scoring = 'r2'))))

plt.scatter(result['AGB_viv'], result['pred'])

plt.scatter(y_test['AGB_viv'], y_pred)
plt.axline([0, 0], slope=1)
plt.show()
result = permutation_importance(
    regressor, X_test, y_test['AGB_viv'], n_repeats=10, random_state=42, n_jobs=2
)
forest_importances = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

