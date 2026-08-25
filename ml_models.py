import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib 

# load dataset from CSV file
hea_df = pd.read_csv("compList_with_descriptors_30May2024.csv")

# display the first five rows
print(hea_df.head())

# display dataset information and data types 
print(hea_df.info())

# remove unnecessary index columns 
hea_df.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)

# check the number of duplicated rows 
print(hea_df.duplicated().sum())

# check the number of missing values in each column 
print(hea_df.isnull().sum())

############################ Removal of Outlier using IQR ############################
# get all input features column names 
X_variables = hea_df.drop(columns = ['YS(Mpa)', 'UTS(Mpa)', 'El(%)'], axis = 1).columns.tolist()

# calculate the first quartile 
Q1 = hea_df[X_variables].quantile(0.25)

# calculate the third quartile 
Q3 = hea_df[X_variables].quantile(0.75)

# calculate the interquartile range 
IQR = Q3 - Q1

# define the lower limit for outlier detection 
lower_bound = Q1 - 1 * IQR

# define the upper limit for outlier deetection 
upper_bound = Q3 + 1 * IQR

# identify the rows containing the outliers
outlier_mask = ((hea_df[X_variables] < lower_bound) | (hea_df[X_variables] > upper_bound)).any(axis = 1)

# remove rows containing outliers and reset index 
hea_df = hea_df[~outlier_mask].reset_index(drop = True)

# display the number of removed outlier rows 
print("Number of Outlier Rows: {}".format(outlier_mask.sum()))

############################ Feature Selection using Lasso ############################
# separate input features and target variables
X_features = hea_df.drop(columns = ['YS(Mpa)', 'UTS(Mpa)', 'El(%)'], axis = 1)
y_targets = hea_df[['YS(Mpa)', 'UTS(Mpa)', 'El(%)']]

# scale features between 0 and 1 before lasso 
scaler_lasso = MinMaxScaler()
x_scaler_lasso = scaler_lasso.fit_transform(X_features)

# create multitask lasso with 5-fold cv
lasso_cv = MultiTaskLassoCV(cv = 5, random_state = 42)

# train the lasso model
lasso_cv.fit(x_scaler_lasso, y_targets)

# obtain feature coefficients from the model 
coefficients = lasso_cv.coef_

# identify features with at least one non-zero coefficients 
selected_mask = np.any(coefficients != 0, axis=0)

# get names of selected features 
selected_features = X_features.columns[selected_mask].tolist()

# get names of dropped features 
dropped_features = X_features.columns[~selected_mask].tolist()

# display selected and dropped features 
print("Selected Features: {}".format(selected_features))
print("Dropped Features: {}".format(dropped_features))

# keep on selected features and target variables 
hea_df = pd.concat([hea_df[selected_features], hea_df[['YS(Mpa)', 'UTS(Mpa)', 'El(%)']]], axis = 1)

############################ Train-Test Split ############################
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hea_df.drop(columns = ['YS(Mpa)', 'UTS(Mpa)', 'El(%)'], axis = 1), 
                                                    hea_df[['YS(Mpa)', 'UTS(Mpa)', 'El(%)']], 
                                                    test_size = 0.25, 
                                                    random_state = 42)

# display the shapes of training and testing datasets
print("\nX-train shape: {}".format(X_train.shape))
print("X-test shape: {}".format(X_test.shape))
print("y-train shape: {}".format(y_train.shape))
print("y-test shape: {}".format(y_test.shape))

############################ Feature Scaling ############################
# get the names of selected input features 
X_features2 = hea_df.drop(columns = ['YS(Mpa)', 'UTS(Mpa)', 'El(%)'], axis = 1).columns

# create a min-max scaler 
scaler2 = MinMaxScaler()

# fit the scaler on training data and transform it 
X_train[X_features2] = scaler2.fit_transform(X_train[X_features2])

# transform testing data using the same scaler 
X_test[X_features2] = scaler2.transform(X_test[X_features2])

############################ Model Training ############################
# create 5-fold cross validation 
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)

# create a multioutput gradient boosting model 
gb_model = MultiOutputRegressor(GradientBoostingRegressor(random_state = 42))

# define hyperparameters for grid search 
param_grid = {'estimator__n_estimators': [50, 100, 150, 200], 
              'estimator__learning_rate': [0.1, 0.2, 0.3], 
              'estimator__max_depth': [2, 3, 4]}

# search for the best hyperparameter combination 
grid_search = GridSearchCV(estimator = gb_model, 
                           param_grid = param_grid, 
                           cv = kfold, 
                           scoring = 'neg_mean_squared_error', 
                           n_jobs = 1, 
                           verbose = 3, 
                           return_train_score = True)

# train models using grid search 
grid_search.fit(X_train, y_train)

# display the best hyperparameters 
print("\nBest Parameters: ")
print(grid_search.best_params_)

# display the best cross validation scores 
print("\nBest score:")
print(grid_search.best_score_)

# retrieve the best model 
best_model = grid_search.best_estimator_

############################ Model Evaluation ############################
# generate predictions for training and testing sets 
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

# extract yield strength, ultimate tensile strength, elongation train predictions
train_pred_ys = train_pred[:, 0]
train_pred_uts = train_pred[:, 1]
train_pred_el = train_pred[:, 2]

# extract yield strength, ultimate tensile strength, elongation test predictions
test_pred_ys = test_pred[:, 0]
test_pred_uts = test_pred[:, 1]
test_pred_el = test_pred[:, 2]

############################ R2 Score ############################
# calculate training R2 scores 
train_ys_r2 = r2_score(y_train['YS(Mpa)'], train_pred_ys)
train_uts_r2 = r2_score(y_train['UTS(Mpa)'], train_pred_uts)
train_el_r2 = r2_score(y_train['El(%)'], train_pred_el)

# calculate testing R2 scores 
test_ys_r2 = r2_score(y_test['YS(Mpa)'], test_pred_ys)
test_uts_r2 = r2_score(y_test['UTS(Mpa)'], test_pred_uts)
test_el_r2 = r2_score(y_test['El(%)'], test_pred_el)

############################ RMSE ############################
# calculate training RMSE scores 
train_ys_rmse = np.sqrt(mean_squared_error(y_train['YS(Mpa)'], train_pred_ys))
train_uts_rmse = np.sqrt(mean_squared_error(y_train['UTS(Mpa)'], train_pred_uts))
train_el_rmse = np.sqrt(mean_squared_error(y_train['El(%)'], train_pred_el))

# calculate testing RMSE scores 
test_ys_rmse = np.sqrt(mean_squared_error(y_test['YS(Mpa)'], test_pred_ys))
test_uts_rmse = np.sqrt(mean_squared_error(y_test['UTS(Mpa)'], test_pred_uts))
test_el_rmse = np.sqrt(mean_squared_error(y_test['El(%)'], test_pred_el))

############################ Display Model Performance ############################
print("\n*********** Yield Strength (YS) ***********")
print("TRAIN: ")
print("R2-Score: {:.4f}".format(train_ys_r2))
print("RMSE: {:.4f}".format(train_ys_rmse))
print()
print("TEST: ")
print("R2-Score: {:.4f}".format(test_ys_r2))
print("RMSE: {:.4f}".format(test_ys_rmse))
print()

print("*********** Ultimate Tensile Strength (UTS) ***********")
print("TRAIN: ")
print("R2-Score: {:.4f}".format(train_uts_r2))
print("RMSE: {:.4f}".format(train_uts_rmse))
print()
print("TEST: ")
print("R2-Score: {:.4f}".format(test_uts_r2))
print("RMSE: {:.4f}".format(test_uts_rmse))
print()

print("*********** Elongation (EL) ***********")
print("TRAIN: ")
print("R2-Score: {:.4f}".format(train_el_r2))
print("RMSE: {:.4f}".format(train_el_rmse))
print()
print("TEST: ")
print("R2-Score: {:.4f}".format(test_el_r2))
print("RMSE: {:.4f}".format(test_el_rmse))


############################ plot YS results  ############################
# plot of yield strength of actual vs predicted on train and test sets 
ys_train_df = pd.DataFrame()
ys_test_df = pd.DataFrame()
ys_train_df['train_actual'] = y_train['YS(Mpa)']
ys_train_df['train_pred'] = train_pred_ys
ys_test_df['test_actual'] = y_test['YS(Mpa)']
ys_test_df['test_pred'] = test_pred_ys

plt.scatter(ys_train_df['train_actual'], ys_train_df['train_pred'], marker = "s", color = "black", label = "Train Set (R2: {:.2f} | RMSE: {:.2f})".format(train_ys_r2, train_ys_rmse))
plt.scatter(ys_test_df['test_actual'], ys_test_df['test_pred'], marker = "D", color = "red", label = "Test Set (R2: {:.2f} | RMSE: {:.2f})".format(test_ys_r2, test_ys_rmse))
plt.xlabel("Actual YS (MPa)", fontweight = 'bold')
plt.ylabel("Predicted YS (MPa)", fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.savefig("yield_strength.png")
plt.close()


############################ plot UTS results  ############################
# plot of ultimate tensile strength of actual vs predicted on train and test sets 
uts_train_df = pd.DataFrame()
uts_test_df = pd.DataFrame()
uts_train_df['train_actual'] = y_train['UTS(Mpa)']
uts_train_df['train_pred'] = train_pred_uts
uts_test_df['test_actual'] = y_test['UTS(Mpa)']
uts_test_df['test_pred'] = test_pred_uts

plt.scatter(uts_train_df['train_actual'], uts_train_df['train_pred'], marker = "s", color = "black", label = "Train Set (R2: {:.2f} | RMSE: {:.2f})".format(train_uts_r2, train_uts_rmse))
plt.scatter(uts_test_df['test_actual'], uts_test_df['test_pred'], marker = "D", color = "red", label = "Test Set (R2: {:.2f} | RMSE: {:.2f})".format(test_uts_r2, test_uts_rmse))
plt.xlabel("Actual UTS (MPa)", fontweight = 'bold')
plt.ylabel("Predicted UTS (MPa)", fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.savefig("ultimate_tensile_strength.png")
plt.close()

############################ plot EL results  ############################
# plot of elongation of actual vs predicted on train and test sets 
el_train_df = pd.DataFrame()
el_test_df = pd.DataFrame()
el_train_df['train_actual'] = y_train['El(%)']
el_train_df['train_pred'] = train_pred_el
el_test_df['test_actual'] = y_test['El(%)']
el_test_df['test_pred'] = test_pred_el

plt.scatter(el_train_df['train_actual'], el_train_df['train_pred'], marker = "s", color = "black", label = "Train Set (R2: {:.2f} | RMSE: {:.2f})".format(train_el_r2, train_el_rmse))
plt.scatter(el_test_df['test_actual'], el_test_df['test_pred'], marker = "D", color = "red", label = "Test Set (R2: {:.2f} | RMSE: {:.2f})".format(test_el_r2, test_el_rmse))
plt.xlabel("Actual EL (%)", fontweight = 'bold')
plt.ylabel("Predicted EL (%)", fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.savefig("elongation.png")
plt.close()

############################ save the best models  ############################
# save model and scaler
joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler2, "scaler.joblib")


print(hea_df.iloc[:, :10].describe())
