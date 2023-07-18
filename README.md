# Project-Dry-Beans-Classification

# Project Report: Dry Beans Classification

This is a project report for the classification of dry beans dataset using basic supervised learning methods. The project involves dataset exploration, data preprocessing, model training, model evaluation, and performance comparison.

## 1. Dataset Exploration

In this project, we used the "dry-beans.csv" dataset, which contains various features of dry beans. Here are the initial steps we took to explore the dataset:

- Loaded the dataset using the pandas library: `df = pd.read_csv('dry-beans.csv')`
- Displayed the first few rows of the dataset: `df.head()`
- Checked the dimensions of the dataset (number of rows and columns): `df.shape`
- Checked the data types of the columns: `df.dtypes`
- Checked for missing values: `df.isnull().sum()`
- Computed descriptive statistics of the dataset: `df.describe()`
- Examined the class distribution of the target variable: `df['Class'].value_counts()`

## 2. Data Preprocessing

To prepare the data for model training, we performed the following preprocessing steps:

- Separated the features (X) and the target variable (y) from the dataset: 
```python
X = df.drop('Class', axis=1)
y = df['Class']
```
- Handled missing values by replacing them with the column means: `X = X.fillna(X.mean())`
- Scaled the numerical features using the StandardScaler: 
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- Encoded the categorical target variable using LabelEncoder: 
```python
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```
- Split the data into training and testing sets for model evaluation: 
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
```

## 3. Model Training

For classification, we trained the following models using the training data:

- Logistic Regression: `lr = LogisticRegression()`
- Decision Tree: `dt = DecisionTreeClassifier()`
- k-Nearest Neighbors: `knn = KNeighborsClassifier()`
- Naïve Bayes: `nb = GaussianNB()`
- Support Vector Machine: `svm = SVC()`

We fit the models on the training data using the `fit()` method of each model.

## 4. Model Evaluation

To evaluate the trained models, we made predictions on the testing data and calculated the accuracy scores using the `accuracy_score` metric. Here are the accuracy scores for each model:

- Logistic Regression Accuracy: (accuracy score)
- Decision Tree Accuracy: (accuracy score)
- k-Nearest Neighbors Accuracy: (accuracy score)
- Naïve Bayes Accuracy: (accuracy score)
- Support Vector Machine Accuracy: (accuracy score)

## 5. Performance Comparison

We compared the performances of the different models using a bar chart visualization. The bar chart displays the accuracy scores of each model, allowing for an easy comparison of their performance.

![Model Performance Comparison](image_link)

In the bar chart, the x-axis represents the models, and the y-axis represents the accuracy. The higher the bar, the better the accuracy.

Overall, the project involved exploring the dataset, preprocessing the data, training multiple classification models, evaluating their performances, and comparing their accuracy scores. The results can be used to determine the best-performing model for the classification of dry beans.

Note: Replace "image_link" in the markdown with the actual image link or attach the image separately in the README file.

**Dataset Source:**
The dry beans dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset).

**References:**
If any references were used during the project, please include them here.

