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

## References
[1] Wang, J., Kwon, H., Kim, H. S., & Lee, B.-J. (2023). A neural network model for high entropy alloy design. Npj Computational Materials, 9(1). https://doi.org/10.1038/s41524-023-01010-x
