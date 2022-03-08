# LifeInsuranceSales
Predicting Agent Bonus 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt



print('Numpy Version',np.__version__)
print('Pandas Version',pd.__version__)
print('Seaborn Version',sns.__version__)
print('Matplotlib Version',matplotlib.__version__)

pip install folium

sns.set()
sns.set_palette(palette='deep')
import folium
from folium.plugins import FastMarkerCluster

df = pd.read_excel('Sales.xlsx',sheet_name = 'Sales')

df.head()

df.shape

df.info()

df.isnull().sum()

df_na = df.isna().sum()
df_na[df_na.values > 0].sort_values(ascending=False)

cat=[]
num=[]
for i in df.columns:
    if df[i].dtype=="object":
        cat.append(i)
    else:
        num.append(i)
print(cat) 
print(num)

#Outliers count
Q1 = df[num].quantile(0.25)
Q3 = df[num].quantile(0.75)
IQR = Q3 - Q1
((df[num] < (Q1 - 1.5 * IQR)) | (df[num] > (Q3 + 1.5 * IQR))).sum()

df['Age'] = df['Age'].fillna(df['Age'].mode()[0])

df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].mode()[0])

df['CustTenure'] = df['CustTenure'].fillna(df['CustTenure'].mode()[0])

df['ExistingPolicyTenure'] = df['ExistingPolicyTenure'].fillna(df['ExistingPolicyTenure'].mode()[0])

df['SumAssured'] = df['SumAssured'].fillna(df['SumAssured'].mode()[0])

df['CustCareScore'] = df['CustCareScore'].fillna(df['CustCareScore'].mean())

df['NumberOfPolicy'] = df['NumberOfPolicy'].fillna(df['NumberOfPolicy'].mean())

df_na = df.isna().sum()
df_na[df_na.values > 0].sort_values(ascending=False)

df.duplicated().sum()

df.describe().T

df[df.Complaint > 0]

for column in df.columns:
    if df[column].dtype == 'object':
        print(column.upper(),': ',df[column].nunique())
        print(df[column].value_counts().sort_values())
        print('\n')

df['Gender']=np.where(df['Gender'] =='Fe male', 'Female', df['Gender'])

df['Occupation']=np.where(df['Occupation'] =='Laarge Business', 'Large Business', df['Occupation'])

df.hist(figsize=(20,30));

sns.pairplot(df.drop('CustID',axis=1),hue='AgentBonus', diag_kind = 'kde');

df.drop('CustID',axis=1).to_excel('cleanData1.xlsx')

def univariateAnalysis_numeric(column,nbins):
    print("Description of " + column)
    print("----------------------------------------------------------------------------\n")
    print(df[column].describe(),end=' ')
    print('\n')
    print("----------------------------------------------------------------------------\n")
    print("Skew for {} is {}".format(column, df[column].skew()))
    print('\n')
    
    
    plt.figure()
    print("Distribution of " + column)
    print("----------------------------------------------------------------------------")
    sns.displot(df[column], kde=False, color='g');
    plt.show()
    
    plt.figure()
    print("BoxPlot of " + column)
    print("----------------------------------------------------------------------------")
    ax = sns.boxplot(x=df[column])
    plt.show()

df_num = df.select_dtypes(include = ['float64', 'int64'])
lstnumericcolumns = list(df_num.columns.values)
for x in lstnumericcolumns:
    univariateAnalysis_numeric(x,20)

plt.figure(figsize=(15,12))
plt.pie(df['Channel'].value_counts(), labels=df['Channel'].value_counts().index, autopct='%1.2f%%', explode=(0,0,0));

plt.figure(figsize=(15,12))
plt.pie(df['EducationField'].value_counts(), labels=df['EducationField'].value_counts().index, autopct='%1.2f%%', explode=(0.1,0,0,0,0,0,0));

plt.figure(figsize=(15,12))
plt.pie(df['Designation'].value_counts(), labels=df['Designation'].value_counts().index, autopct='%1.2f%%', explode=(0,0,0,0,0,0));

sns.barplot(df.Gender.value_counts().index,df.Gender.value_counts().values);

sns.barplot(df.MaritalStatus.value_counts().index,df.MaritalStatus.value_counts().values);

sns.barplot(df.Zone.value_counts().index,df.Zone.value_counts().values);

sns.barplot(df.PaymentMethod.value_counts().index,df.PaymentMethod.value_counts().values);

plt.figure(figsize=(15,8))
sns.barplot(df["EducationField"],df["MonthlyIncome"])

plt.figure(figsize=(15,8))
sns.barplot(df["Occupation"],df["MonthlyIncome"])

plt.figure(figsize=(15,8))
sns.barplot(df["Designation"],df["MonthlyIncome"])

plt.figure(figsize=(15,8))
sns.barplot(df["Designation"],df["SumAssured"])

plt.figure(figsize=(15,8))
sns.barplot(df["Zone"],df["AgentBonus"])

plt.figure(figsize=(15,8))
sns.barplot(df["Designation"],df["AgentBonus"])

plt.figure(figsize=(15,8))
sns.barplot(df["Gender"],df["AgentBonus"])

plt.scatter(df.MonthlyIncome, df.SumAssured)

plt.scatter(df.MonthlyIncome, df.AgentBonus)

corr = df.drop('CustID',axis=1).corr()
round(corr,2)
fig_dims = (10, 5)
fig = plt.subplots(figsize=fig_dims)
mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
sns.heatmap(round(corr,2), annot=True, mask=mask)

#Outliers count
Q1 = df[num].quantile(0.25)
Q3 = df[num].quantile(0.75)
IQR = Q3 - Q1
((df[num] < (Q1 - 1.5 * IQR)) | (df[num] > (Q3 + 1.5 * IQR))).sum()

def getRanges(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range

lr,ur=getRanges(df['AgentBonus'])
print("lower range",lr, "and upper range", ur)
df['AgentBonus']=np.where(df['AgentBonus']>ur,ur,df['AgentBonus'])
df['AgentBonus']=np.where(df['AgentBonus']<lr,lr,df['AgentBonus'])

lr,ur=getRanges(df['Age'])
print("lower range",lr, "and upper range", ur)
df['Age']=np.where(df['Age']>ur,ur,df['Age'])
df['Age']=np.where(df['Age']<lr,lr,df['Age'])

lr,ur=getRanges(df['CustTenure'])
print("lower range",lr, "and upper range", ur)
df['CustTenure']=np.where(df['CustTenure']>ur,ur,df['CustTenure'])
df['CustTenure']=np.where(df['CustTenure']<lr,lr,df['CustTenure'])

lr,ur=getRanges(df['ExistingProdType'])
print("lower range",lr, "and upper range", ur)
df['ExistingProdType']=np.where(df['ExistingProdType']>ur,ur,df['ExistingProdType'])
df['ExistingProdType']=np.where(df['ExistingProdType']<lr,lr,df['ExistingProdType'])

lr,ur=getRanges(df['MonthlyIncome'])
print("lower range",lr, "and upper range", ur)
df['MonthlyIncome']=np.where(df['MonthlyIncome']>ur,ur,df['MonthlyIncome'])
df['MonthlyIncome']=np.where(df['MonthlyIncome']<lr,lr,df['MonthlyIncome'])

lr,ur=getRanges(df['ExistingPolicyTenure'])
print("lower range",lr, "and upper range", ur)
df['ExistingPolicyTenure']=np.where(df['ExistingPolicyTenure']>ur,ur,df['ExistingPolicyTenure'])
df['ExistingPolicyTenure']=np.where(df['ExistingPolicyTenure']<lr,lr,df['ExistingPolicyTenure'])

lr,ur=getRanges(df['SumAssured'])
print("lower range",lr, "and upper range", ur)
df['SumAssured']=np.where(df['SumAssured']>ur,ur,df['SumAssured'])
df['SumAssured']=np.where(df['SumAssured']<lr,lr,df['SumAssured'])

lr,ur=getRanges(df['LastMonthCalls'])
print("lower range",lr, "and upper range", ur)
df['LastMonthCalls']=np.where(df['LastMonthCalls']>ur,ur,df['LastMonthCalls'])
df['LastMonthCalls']=np.where(df['LastMonthCalls']<lr,lr,df['LastMonthCalls'])

#Outliers count
Q1 = df[num].quantile(0.25)
Q3 = df[num].quantile(0.75)
IQR = Q3 - Q1
((df[num] < (Q1 - 1.5 * IQR)) | (df[num] > (Q3 + 1.5 * IQR))).sum()

df.drop('CustID',axis=1,inplace=True)

df_new =pd.get_dummies(df, columns=cat,drop_first=True)

df_new.head()

df_new['Bonus_ratio'] = df_new['AgentBonus']/df_new['SumAssured']

df_new.head()

# Copy all the predictor variables into X dataframe
X = df_new.drop('AgentBonus', axis=1)

# Copy target into the y dataframe.  
y = df_new['AgentBonus']

from scipy.stats import zscore
data_scaled=X.apply(zscore)

data_scaled.head()

data_scaled.describe().T

from sklearn.cluster import KMeans 

k_means = KMeans(n_clusters = 2)
k_means.fit(data_scaled)
k_means.inertia_

wss = []
for i in range(1,11):
    KM = KMeans(n_clusters=i)
    KM.fit(data_scaled)
    wss.append(KM.inertia_)
    print('wss for '+ str(i)+ ' clusters is : ' +str(KM.inertia_))

plt.plot(range(1,11), wss)

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
sc = StandardScaler()

df_normalized = normalize(data_scaled)
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_normalized)
df_pca = pd.DataFrame(df_pca)
df_pca.columns = ['P1', 'P2', 'P3']


km = KMeans(n_clusters = 3)
plt.figure(figsize =(8, 8))
plt.scatter(df_pca['P1'], df_pca['P2'], c = km.fit_predict(df_pca), cmap ='rainbow')
plt.title("K-means  Clusters - Scatter Plot", fontsize=18)
plt.show()

df_new.to_csv('cleanData.csv')

from sklearn.metrics import mean_squared_error, mean_absolute_error

import sklearn
print(sklearn.__version__)

data = df_new.copy()

data.head()

data.columns

data = data.rename(columns={"Channel_Third Party Partner" : "Channel_Third_Party_Partner","Occupation_Large Business":"Occupation_Large_Business","Occupation_Small Business":"Occupation_Small_Business","EducationField_Post Graduate":"EducationField_Post_Graduate","EducationField_Under Graduate":"EducationField_Under_Graduate","Designation_Senior Manager":"Designation_Senior_Manager"})

X = data.drop('AgentBonus', axis=1)
y = data[['AgentBonus']]

from sklearn.model_selection import train_test_split

# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=0)

print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)

## Linear Regression

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# coefficients
for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))

#intercept
intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))

#train
regression_model.score(X_train, y_train)

# test
regression_model.score(X_test, y_test)

#train
LMmse = np.mean((regression_model.predict(X_train)-y_train)**2)
LMmse

import math

LMrmse = math.sqrt(LMmse)
LMrmse

#test
LMmse = np.mean((regression_model.predict(X_test)-y_test)**2)
LMmse

LMrmse = math.sqrt(LMmse)
LMrmse

#train
LMmae = mean_absolute_error(y_train,(regression_model.predict(X_train)))
LMmae

#test
LMmae = mean_absolute_error(y_test,(regression_model.predict(X_test)))
LMmae

y_pred = regression_model.predict(X_test)

plt.scatter(y_test['AgentBonus'], y_pred)

## Stats model Linear Regression

data_train = pd.concat([X_train, y_train], axis=1)
data_train.head()

import statsmodels.formula.api as sm
lm = sm.ols(formula= 'AgentBonus ~ Age+CustTenure+ExistingProdType+NumberOfPolicy+MonthlyIncome+Complaint+ExistingPolicyTenure+SumAssured+LastMonthCalls+CustCareScore+Channel_Online+Channel_Third_Party_Partner+Occupation_Large_Business+Occupation_Salaried+Occupation_Small_Business+EducationField_Engineer+EducationField_Graduate+EducationField_MBA+EducationField_Post_Graduate+EducationField_UG+EducationField_Under_Graduate+Gender_Male+Designation_Exe+Designation_Executive+Designation_Manager+Designation_Senior_Manager+Designation_VP+MaritalStatus_Married+MaritalStatus_Single+MaritalStatus_Unmarried+Zone_North+Zone_South+Zone_West+PaymentMethod_Monthly+PaymentMethod_Quarterly+PaymentMethod_Yearly+Bonus_ratio', data = data_train).fit()
lm.params

print(lm.summary())  

lm1 = sm.ols(formula= 'AgentBonus ~ Age+CustTenure+MonthlyIncome+Complaint+SumAssured+Channel_Third_Party_Partner+Designation_Executive+Designation_Manager+Designation_Senior_Manager+PaymentMethod_Yearly+Bonus_ratio', data = data_train).fit()
lm1.params
print(lm1.summary()) 

 

for i,j in np.array(lm.params.reset_index()):
    print('({}) * {} +'.format(round(j,2),i),end=' ')

y_pred = regression_model.predict(X_test)

## Centering data

from scipy.stats import zscore

X_train_scaled  = X_train.apply(zscore)
X_test_scaled = X_test.apply(zscore)
y_train_scaled = y_train.apply(zscore)
y_test_scaled = y_test.apply(zscore)

regression_model = LinearRegression()
regression_model.fit(X_train_scaled, y_train_scaled)

for idx, col_name in enumerate(X_train_scaled.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))

data_train = pd.concat([X_train_scaled, y_train_scaled], axis=1)
data_train.head()

lm = sm.ols(formula= 'AgentBonus ~ Age+CustTenure+ExistingProdType+NumberOfPolicy+MonthlyIncome+Complaint+ExistingPolicyTenure+SumAssured+LastMonthCalls+CustCareScore+Channel_Online+Channel_Third_Party_Partner+Occupation_Large_Business+Occupation_Salaried+Occupation_Small_Business+EducationField_Engineer+EducationField_Graduate+EducationField_MBA+EducationField_Post_Graduate+EducationField_UG+EducationField_Under_Graduate+Gender_Male+Designation_Exe+Designation_Executive+Designation_Manager+Designation_Senior_Manager+Designation_VP+MaritalStatus_Married+MaritalStatus_Single+MaritalStatus_Unmarried+Zone_North+Zone_South+Zone_West+PaymentMethod_Monthly+PaymentMethod_Quarterly+PaymentMethod_Yearly+Bonus_ratio', data = data_train).fit()
lm.params

for i,j in np.array(lm.params.reset_index()):
    print('({}) * {} +'.format(round(j,2),i),end=' ')

lm.params

regression_model.coef_[0]
x=pd.DataFrame(regression_model.coef_[0],index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='dark')
plt.xlabel('Feature Importance in %')
plt.ylabel('Features')
plt.title('Feature Importance using Linear regression')
plt.show()

intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))

regression_model.score(X_train_scaled, y_train_scaled)

regression_model.score(X_test_scaled, y_test_scaled)

mse = np.mean((regression_model.predict(X_train_scaled)-y_train_scaled)**2)

import math

math.sqrt(mse)

mse = np.mean((regression_model.predict(X_test_scaled)-y_test_scaled)**2)

import math

math.sqrt(mse)

y_pred = regression_model.predict(X_test_scaled)

mae = mean_absolute_error(y_train_scaled,(regression_model.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(regression_model.predict(X_test_scaled)))
mae

plt.scatter(y_test_scaled['AgentBonus'], y_pred)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

ridge = Ridge(alpha=.3)
ridge.fit(X_train_scaled,y_train_scaled)
print ("Ridge model:", (ridge.coef_))

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled,y_train_scaled)
print ("Lasso model:", (lasso.coef_))

r_train_acc = ridge.score(X_train_scaled, y_train_scaled)
r_test_acc = ridge.score(X_test_scaled, y_test_scaled)

from sklearn import metrics

predicted_train = ridge.fit(X_train_scaled, y_train_scaled).predict(X_train_scaled)
r_train_mae = metrics.mean_absolute_error(y_train_scaled, predicted_train)
predicted_test = ridge.fit(X_train_scaled, y_train_scaled).predict(X_test_scaled)
r_test_mae = metrics.mean_absolute_error(y_test_scaled, predicted_test)

predicted_train = ridge.fit(X_train_scaled, y_train_scaled).predict(X_train_scaled)
r_train_rmse = np.sqrt(metrics.mean_squared_error(y_train_scaled, predicted_train))
predicted_test = ridge.fit(X_train_scaled, y_train_scaled).predict(X_test_scaled)
r_test_rmse = np.sqrt(metrics.mean_squared_error(y_test_scaled, predicted_test))

index=['Train Accuracy', 'Test Accuracy', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']
data = pd.DataFrame({'Ridge Regression':[r_train_acc,r_test_acc,r_train_rmse,r_test_rmse,r_train_mae,r_test_mae]},index=index)
round(data,2)

l_train_acc = lasso.score(X_train_scaled, y_train_scaled)
l_test_acc = lasso.score(X_test_scaled, y_test_scaled)

predicted_train = lasso.fit(X_train_scaled, y_train_scaled).predict(X_train_scaled)
l_train_mae = metrics.mean_absolute_error(y_train_scaled, predicted_train)
predicted_test = lasso.fit(X_train_scaled, y_train_scaled).predict(X_test_scaled)
l_test_mae = metrics.mean_absolute_error(y_test_scaled, predicted_test)

predicted_train = lasso.fit(X_train_scaled, y_train_scaled).predict(X_train_scaled)
l_train_rmse = np.sqrt(metrics.mean_squared_error(y_train_scaled, predicted_train))
predicted_test = lasso.fit(X_train_scaled, y_train_scaled).predict(X_test_scaled)
l_test_rmse = np.sqrt(metrics.mean_squared_error(y_test_scaled, predicted_test))

index=['Train Accuracy', 'Test Accuracy', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']
data = pd.DataFrame({'Lasso Regression':[l_train_acc,l_test_acc,l_train_rmse,l_test_rmse,l_train_mae,l_test_mae]},index=index)
round(data,2)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])] 

i=0
for column in X.columns:
    if i < 38:
        if(vif[i] > 5):
            print (column ,"--->",  vif[i])
            i = i+1
        
        

## Building a Neural Network Classifier

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(random_state=1)

mlp.fit(X_train_scaled, y_train_scaled)

mlp.score(X_train_scaled, y_train_scaled)

mlp.score(X_test_scaled, y_test_scaled)

from sklearn import metrics

from sklearn.metrics import mean_squared_error

## training data

MLPrmse = mean_squared_error(y_train_scaled,(mlp.predict(X_train_scaled)),squared=False)
MLPrmse

MLPrmse = mean_squared_error(y_test_scaled,(mlp.predict(X_test_scaled)),squared=False)
MLPrmse

mae = mean_absolute_error(y_train_scaled,(mlp.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(mlp.predict(X_test_scaled)))
mae

plt.scatter(y_test_scaled,(mlp.predict(X_test_scaled)))

## Tuning MLPRegressor

param_grid = {
    'hidden_layer_sizes': [100,50,20,10], 
    'max_iter': [2500,5000], 
    'solver': ['adam','sgd'], 
    'tol': [0.01], 
}

from sklearn.model_selection import GridSearchCV

gsc = GridSearchCV(
    mlp,
    param_grid,
    cv=5, verbose=0, n_jobs=-1)

grid_result = gsc.fit(X_train_scaled, y_train_scaled)

best_params = grid_result.best_params_
best_params

best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"], 
                        tol=best_params["tol"],
                        solver=best_params["solver"],
                        max_iter= 5000)

best_mlp.fit(X_train_scaled, y_train_scaled)

best_mlp.score(X_train_scaled, y_train_scaled)

best_mlp.score(X_test_scaled, y_test_scaled)

MLPrmse = mean_squared_error(y_train_scaled,(best_mlp.predict(X_train_scaled)),squared=False)
MLPrmse

MLPrmse = mean_squared_error(y_test_scaled,(best_mlp.predict(X_test_scaled)),squared=False)
MLPrmse

mae = mean_absolute_error(y_train_scaled,(best_mlp.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(best_mlp.predict(X_test_scaled)))
mae

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(y_test_scaled, y_test_scaled)
plt.scatter(y_test_scaled, (best_mlp.predict(X_test_scaled)), s=10, c="yellow")
plt.title("Actual vs Predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

## Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

dtR = DecisionTreeRegressor(random_state=0)

dtR.fit(X_train_scaled, y_train_scaled)

x=pd.DataFrame(dtR.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='dark')
plt.xlabel('Feature Importance in %')
plt.ylabel('Features')
plt.title('Feature Importance using Decision Tree Regressor')
plt.show()

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(y_test_scaled, y_test_scaled)
plt.scatter(y_test_scaled, (dtR.predict(X_test_scaled)), s=10, c="yellow")
plt.title("Actual vs Predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

dtR.score(X_train_scaled, y_train_scaled)

dtR.score(X_test_scaled, y_test_scaled)

dtRmse = mean_squared_error(y_train_scaled,(dtR.predict(X_train_scaled)),squared=False)
dtRmse

dtRrmse = mean_squared_error(y_test_scaled,(dtR.predict(X_test_scaled)),squared=False)
dtRrmse

mae = mean_absolute_error(y_train_scaled,(dtR.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(dtR.predict(X_test_scaled)))
mae

## Pruning Decision tree

param_grid = {
    
    'max_depth': [10,20,30,50],
    'min_samples_leaf': [50,100,150], 
    'min_samples_split': [50,100,150,300,450],
}

dtcl = DecisionTreeRegressor(random_state=1)

grid_search = GridSearchCV(estimator = dtcl, param_grid = param_grid, cv = 10)

grid_search.fit(X_train_scaled, y_train_scaled)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
best_grid

from sklearn import tree

train_char_label = ['No', 'Yes']
Tree_File = open('tree4.dot','w')
dot_data = tree.export_graphviz(best_grid, out_file=Tree_File, feature_names = list(X_train), class_names = list(train_char_label))

Tree_File.close()

best_grid.score(X_train_scaled, y_train_scaled)

best_grid.score(X_test_scaled, y_test_scaled)

dtRmse = mean_squared_error(y_train_scaled,(best_grid.predict(X_train_scaled)),squared=False)
dtRmse

dtRrmse = mean_squared_error(y_test_scaled,(best_grid.predict(X_test_scaled)),squared=False)
dtRrmse

mae = mean_absolute_error(y_train_scaled,(best_grid.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(best_grid.predict(X_test_scaled)))
mae

x=pd.DataFrame(best_grid.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='dark')
plt.xlabel('Feature Importance in %')
plt.ylabel('Features')
plt.title('Feature Importance using Decision Tree Regressor')
plt.show()

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(y_test_scaled, y_test_scaled)
plt.scatter(y_test_scaled, (best_grid.predict(X_test_scaled)), s=10, c="yellow")
plt.title("Actual vs Predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

## Random Forest Regressor



from sklearn.ensemble import RandomForestRegressor

rfR = RandomForestRegressor(random_state=0)

rfR.fit(X_train_scaled, y_train_scaled)

rfR.score(X_train_scaled, y_train_scaled)

rfR.score(X_test_scaled, y_test_scaled)

rfRmse = mean_squared_error(y_train_scaled,(rfR.predict(X_train_scaled)),squared=False)
rfRmse

rfRrmse = mean_squared_error(y_test_scaled,(rfR.predict(X_test_scaled)),squared=True)
rfRrmse

mae = mean_absolute_error(y_train_scaled,(rfR.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(rfR.predict(X_test_scaled)))
mae

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(y_test_scaled, y_test_scaled)
plt.scatter(y_test_scaled, (rfR.predict(X_test_scaled)), s=10, c="yellow")
plt.title("Actual vs Predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

x=pd.DataFrame(rfR.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='dark')
plt.xlabel('Feature Importance in %')
plt.ylabel('Features')
plt.title('Feature Importance using Random Forest')
plt.show()

## Tuning parameters

param_grid = {
    'max_depth': [10,20,30,40],
    'max_features': [6,7,8,9],
    'min_samples_leaf': [10,50,100],
    'min_samples_split': [50,60,70], 
    'n_estimators': [100,200,300]
}


rfcl = RandomForestRegressor(random_state=1)

grid_search = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 5)

grid_search.fit(X_train_scaled, y_train_scaled)

grid_search.best_params_

best_grid = grid_search.best_estimator_

best_grid

best_grid.score(X_train_scaled, y_train_scaled)

best_grid.score(X_test_scaled, y_test_scaled)

rFRmse = mean_squared_error(y_train_scaled,(best_grid.predict(X_train_scaled)),squared=False)
rFRmse

rFRmse = mean_squared_error(y_test_scaled,(best_grid.predict(X_test_scaled)),squared=False)
rFRmse

mae = mean_absolute_error(y_train_scaled,(best_grid.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(best_grid.predict(X_test_scaled)))
mae

x=pd.DataFrame(best_grid.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='dark')
plt.xlabel('Feature Importance in %')
plt.ylabel('Features')
plt.title('Feature Importance using Random Forest Regressor')
plt.show()

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(y_test_scaled, y_test_scaled)
plt.scatter(y_test_scaled, (best_grid.predict(X_test_scaled)), s=10, c="yellow")
plt.title("Actual vs Predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

from sklearn.ensemble import GradientBoostingRegressor

gbcl = GradientBoostingRegressor(random_state=1)
gbcl = gbcl.fit(X_train_scaled, y_train_scaled)

y_train_predict = gbcl.predict(X_train_scaled)

gbcl.score(X_train_scaled, y_train_scaled)

gbcl.score(X_test_scaled, y_test_scaled)

gbRmse = mean_squared_error(y_train_scaled,(gbcl.predict(X_train_scaled)),squared=False)
gbRmse

gbRmse = mean_squared_error(y_test_scaled,(gbcl.predict(X_test_scaled)),squared=False)
gbRmse

mae = mean_absolute_error(y_train_scaled,(gbcl.predict(X_train_scaled)))
mae

mae = mean_absolute_error(y_test_scaled,(gbcl.predict(X_test_scaled)))
mae

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(y_test_scaled, y_test_scaled)
plt.scatter(y_test_scaled, (gbcl.predict(X_test_scaled)), s=10, c="yellow")
plt.title("Actual vs Predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()



