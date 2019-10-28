# EPFL-Machine-Learning-Higgs-2019

Higgs Boson Machine Learning Challenge: binary classification problem to distinguish whether the observed decay particles pertain to a Higgs boson or not.

#### Prerequisites

* Python 3.6
* Numpy 

#### How to run

Create an account in AIcrowd, download the train.csv and test.csv from https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files and put the files in a folder named "data".

```python3 run.py```

#### File descriptions

implementations.py - contains numpy implementations of the following methods: 
* least squares (using gradient descent, stochastic gradient descent, and normal equations)
* ridge regression using normal equations
* logistic regression
* regularized logistic regression

preprocessing_data.py - contains util methods for preprocessing data:
* standardizing data
* oversampling/undersampling classes of data
* augmenting data using functions such as logarithm, polynomials, and square root
* handling outliers

helpers.py - contains util methods for 
* loading data
* visualizing data
* analyzing model performance
* making csv submission


