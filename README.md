<img width="211" height="101" alt="images" src="https://github.com/user-attachments/assets/68a3f812-e1f2-4c38-8ed9-a7587cff0788" />

#### _A*STAR Institute of High Performance Computing (IHPC) ARIA Internship Project on Materials Informatics 2024_


# Docker-Based Deployment of a Streamlit Application for HEA Property Prediction

<img width="1000" height="450" alt="high-entropy-alloy" src="https://github.com/user-attachments/assets/af119e96-8726-4e78-9171-e030379ffdb6" />

## 1. Introduction 
High-entropy alloys (HEAs) are advanced metallic materials containing multiple principal elements and have attracted significant interest because of their potential for achieving desirable combinations of mechanical properties. However, experimentally determining mechanical properties for a large number of possible HEA compositions can be time-consuming and resource-intensive. Machine learning (ML) provides a data-driven approach for identifying relationships between material descriptors and experimentally measured mechanical properties.

This study developed a machine learning pipeline to predict three important mechanical properties of high-entropy alloys: yield strength (YS), ultimate tensile strength (UTS), and elongation (El). The Python implementation uses data preprocessing, outlier removal, multi-task Lasso feature selection, feature scaling, multi-output Gradient Boosting Regression, hyperparameter optimisation, and model evaluation using the coefficient of determination (R²) and root mean square error (RMSE).

The overall workflow implemented in the program consists of dataset preparation, data cleaning, outlier removal, feature selection, train-test splitting, feature scaling, model optimisation, prediction, performance evaluation, visualisation, and model serialisation.

## 2. Dataset and Target Variables
The dataset is loaded from the CSV file compList_with_descriptors_30May2024.csv. The program initially examines the first five records and dataset information to inspect the structure and data types.

An unnecessary index column named Unnamed: 0 is removed before modelling. Duplicate records and missing values are also checked. These steps are important because duplicated observations can bias the learning process, while missing values can prevent successful model training.

The three response variables are explicitly defined as:

- YS (MPa): Yield Strength
- UTS (MPa): Ultimate Tensile Strength
- El (%): Elongation

All remaining columns are initially treated as input features. Therefore, the model attempts to learn the relationship between the available HEA composition/material descriptors and the three mechanical properties simultaneously.

```python
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
```

## 3. Data Preprocessing
3.1 Removal of Outliers

The code applies an interquartile range (IQR)-based method to identify anomalous observations among the input features.

For each input feature, the first quartile (Q1), third quartile (Q3), and IQR are calculated:

IQR = Q3 − Q1

The lower and upper boundaries are defined as:

Lower Bound = Q1 − 1 × IQR

Upper Bound = Q3 + 1 × IQR

Unlike the more commonly used 1.5 × IQR criterion, this implementation deliberately uses 1 × IQR, resulting in a more stringent definition of feature-level outliers.

A row is classified as an outlier when at least one input feature falls outside its corresponding lower or upper boundary. All such rows are subsequently removed and the dataframe index is reset. The program also reports the number of rows removed during this stage.

This preprocessing step is particularly relevant to materials informatics because extreme descriptor values may have a substantial influence on tree-based regression models. However, because the threshold is relatively strict, it may also remove legitimate HEA compositions representing unusual but physically meaningful material behaviour.

```python
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
```
## 4. Feature Selection Using Multi-Task Lasso

Following outlier removal, the program separates the input features from the three target variables.

A MinMaxScaler is first applied to transform all input features into the range between 0 and 1. This scaling is performed before Lasso because regularisation-based methods are sensitive to differences in feature magnitude.

The use of MultiTask Lasso is appropriate for this problem because the study predicts three related mechanical properties simultaneously. Instead of selecting features independently for YS, UTS, and elongation, the method identifies features that contribute to at least one of the target variables.

The model uses five-fold cross-validation to determine the regularisation strength. After training, the coefficient matrix is examined. A feature is retained when at least one of its coefficients is non-zero for the three target properties.

Consequently, the feature-selection procedure produces two groups:

- Selected features with at least one non-zero coefficient.
- Dropped features whose coefficients are zero for all three targets.

Only the selected features are retained for subsequent model development.

This step reduces the dimensionality of the HEA descriptor space and potentially removes descriptors that contribute little to predicting the mechanical properties.

```python
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
```
## 5. Training and Testing Dataset

After feature selection, the dataset is divided into training and testing subsets using a 75:25 split.

The three mechanical properties remain as a multi-output target dataframe. This ensures that YS, UTS, and elongation are evaluated for the same observations in the testing set.

The use of a fixed random state improves reproducibility because the same observations will be assigned to the training and testing sets when the program is rerun.

```python
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
```

## 6. Feature Scaling

A second MinMaxScaler is created for the selected features. 

This is an appropriate approach because the testing dataset should not influence the estimation of preprocessing parameters. The same fitted scaler can also later be used to transform new HEA data before making predictions.

```python
# get the names of selected input features 
X_features2 = hea_df.drop(columns = ['YS(Mpa)', 'UTS(Mpa)', 'El(%)'], axis = 1).columns

# create a min-max scaler 
scaler2 = MinMaxScaler()

# fit the scaler on training data and transform it 
X_train[X_features2] = scaler2.fit_transform(X_train[X_features2])

# transform testing data using the same scaler 
X_test[X_features2] = scaler2.transform(X_test[X_features2])
```

## 7. Gradient Boosting Regression Model

The main predictive model is GradientBoostingRegressor. Since the standard Gradient Boosting Regressor predicts a single target, the implementation wraps it using:

MultiOutputRegressor(GradientBoostingRegressor())

This produces separate Gradient Boosting regression models for the three target variables while maintaining a common multi-output prediction interface.

Gradient Boosting is suitable for this application because the relationship between material descriptors and mechanical properties may be nonlinear. The model combines multiple weak regression models sequentially, with each successive estimator attempting to improve the errors made by the previous estimators.

The base Gradient Boosting model is configured with random_state = 42.

```python
# create 5-fold cross validation 
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)

# create a multioutput gradient boosting model 
gb_model = MultiOutputRegressor(GradientBoostingRegressor(random_state = 42))

```

## 8. Hyperparameter Optimisation

GridSearchCV is used to determine an appropriate combination of Gradient Boosting hyperparameters.

Each combination is evaluated using five-fold cross-validation, resulting in multiple model fits during the grid-search process.

The optimisation criterion is negative mean squared error (neg_mean_squared_error). Because scikit-learn represents losses as negative scores when maximising a scoring metric, the grid search selects the configuration associated with the lowest mean squared error.

The best hyperparameters and best cross-validation score are printed after the search. The resulting model is stored as best_model.

```python
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
```

## 9. Model Evaluation

After hyperparameter optimisation, the selected model is used to generate predictions for both the training and testing datasets.

The predictions are separated into three individual outputs:

- Yield Strength prediction
- Ultimate Tensile Strength prediction
- Elongation prediction

Two evaluation metrics are calculated for each property.

```python
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
```

## 10. R² Score

The coefficient of determination, R², measures how well the predicted values explain the variation in the observed values.

The code calculates separate R² values for the training and testing datasets. 

A higher R² indicates stronger predictive agreement between the predicted and experimental values. The testing R² values are particularly important because they measure performance on observations that were not used directly for model fitting.

```python
# calculate training R2 scores 
train_ys_r2 = r2_score(y_train['YS(Mpa)'], train_pred_ys)
train_uts_r2 = r2_score(y_train['UTS(Mpa)'], train_pred_uts)
train_el_r2 = r2_score(y_train['El(%)'], train_pred_el)

# calculate testing R2 scores 
test_ys_r2 = r2_score(y_test['YS(Mpa)'], test_pred_ys)
test_uts_r2 = r2_score(y_test['UTS(Mpa)'], test_pred_uts)
test_el_r2 = r2_score(y_test['El(%)'], test_pred_el)
```

## 11. Root Mean Square Error

RMSE is calculated for each target variable using:

RMSE = √MSE

The code reports training and testing RMSE separately for YS, UTS, and elongation.

Unlike R², RMSE is expressed in the same units as the target variable. Therefore:

- YS RMSE is expressed in MPa.
- UTS RMSE is expressed in MPa.
- Elongation RMSE is expressed in percentage points.

Lower RMSE values indicate smaller prediction errors.

The program prints the resulting performance in a structured format for each mechanical property. However, the numerical output is not included in the uploaded source code, so the exact R² and RMSE values cannot be reported from the code alone.

```python
# calculate training RMSE scores 
train_ys_rmse = np.sqrt(mean_squared_error(y_train['YS(Mpa)'], train_pred_ys))
train_uts_rmse = np.sqrt(mean_squared_error(y_train['UTS(Mpa)'], train_pred_uts))
train_el_rmse = np.sqrt(mean_squared_error(y_train['El(%)'], train_pred_el))

# calculate testing RMSE scores 
test_ys_rmse = np.sqrt(mean_squared_error(y_test['YS(Mpa)'], test_pred_ys))
test_uts_rmse = np.sqrt(mean_squared_error(y_test['UTS(Mpa)'], test_pred_uts))
test_el_rmse = np.sqrt(mean_squared_error(y_test['El(%)'], test_pred_el))
```

## 12. Prediction Visualisation

The program generates three actual-versus-predicted scatter plots.

The first plot compares experimental and predicted yield strength, followed by equivalent plots for ultimate tensile strength and elongation.

For each plot:
- Training observations are displayed separately from testing observations.
- Actual values are placed on the x-axis.
- Predicted values are placed on the y-axis.
- R² and RMSE values are included in the plot legend.

These plots provide a visual assessment of model performance. Points located close to an ideal diagonal relationship between actual and predicted values would indicate good predictive agreement. The separation between training and testing points can also provide an indication of potential overfitting.

## 13. Model Deployment and Reproducibility

The final optimised model and feature scaler are saved using Joblib:

- best_model.joblib
- scaler.joblib

Saving these objects allows the trained model to be reused without repeating the complete training process. For future HEA compositions, the same feature preprocessing procedure can be applied using the saved scaler, after which the saved model can generate predictions for YS, UTS, and elongation.

This provides a foundation for integrating the model into a materials informatics application, such as a Streamlit interface or an automated prediction pipeline.

```python
# save model and scaler
joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler2, "scaler.joblib")
```

## 14. Discussion 

<img width="1402" height="377" alt="Screenshot 2026-08-26 094428" src="https://github.com/user-attachments/assets/3401fb78-b10f-40e0-8047-0c0c537ae5cf" />


The Gradient Boosting model demonstrated strong predictive performance for the three mechanical properties of the high-entropy alloys. The results show a close relationship between the actual and predicted values, with most data points distributed close to the expected diagonal trend.

For yield strength (YS), the model achieved a training R² of 0.99 with an RMSE of 35.13 MPa, while the testing dataset achieved an R² of 0.94 and an RMSE of 106.43 MPa. The high testing R² indicates that the model can explain approximately 94% of the variation in the observed yield strength. Although several testing points show larger deviations from the main trend, the overall prediction accuracy remains strong.

For ultimate tensile strength (UTS), the model produced a training R² of 0.99 and RMSE of 30.13 MPa. On the testing dataset, the R² decreased slightly to 0.91, with an RMSE of 99.98 MPa. The scatter plot shows that most predictions follow the actual UTS values closely. However, several testing observations, particularly at higher UTS values, exhibit noticeable deviations. Nevertheless, an R² of 0.91 demonstrates that the model provides a strong prediction of UTS.

The prediction of elongation (El) was comparatively weaker. The training results achieved an R² of 0.97 and RMSE of 3.85%, whereas the testing results achieved an R² of 0.81 and RMSE of 10.90%. The larger difference between training and testing performance suggests that the model has more difficulty generalising elongation to unseen HEA compositions. Several testing points show substantial deviations from the overall trend, including observations at both low and high elongation values.

Overall, the model performed best for yield strength, followed by ultimate tensile strength, while elongation was the most challenging property to predict. The reduction in R² from training to testing, particularly for elongation, also indicates some degree of overfitting. Nevertheless, the testing R² values of 0.94 for YS, 0.91 for UTS, and 0.81 for elongation demonstrate that the developed machine learning approach can effectively capture the relationships between the selected HEA descriptors and their mechanical properties.



## References
[1] Wang, J., Kwon, H., Kim, H. S., & Lee, B.-J. (2023). A neural network model for high entropy alloy design. Npj Computational Materials, 9(1). https://doi.org/10.1038/s41524-023-01010-x
