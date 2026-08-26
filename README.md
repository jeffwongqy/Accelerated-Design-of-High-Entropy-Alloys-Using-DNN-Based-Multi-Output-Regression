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
## 5. 


## References
[1] Wang, J., Kwon, H., Kim, H. S., & Lee, B.-J. (2023). A neural network model for high entropy alloy design. Npj Computational Materials, 9(1). https://doi.org/10.1038/s41524-023-01010-x
