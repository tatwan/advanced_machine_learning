# Feature Selection with LASSO

In this activity, you'll use a LASSO model to improve a linear regression on the residential building dataset.

## Instructions

* Use the starter file [WranglingFeatures.ipynb](Unsolved/WranglingFeatures.ipynb) and [residential-building.csv](Resources/residential-building.csv) dataset with this activity.

* Fit a linear regression to the imported and split dataset. Print the score of the model.

* Fit a LASSO regression model to the imported and split dataset.

* Import `SelectModel` to extract the best features from the LASSO model.

* Fit a new linear regression to the data with only the selected features (by using `SelectModel`). Print the score of the model.

* Compare the scores of the two linear regression models.

## Reference

Rafiei, M.H. and Adeli, H. (2015). "A Novel Machine Learning Model for Estimation of Sale Prices of Real Estate Units." ASCE, Journal of Construction Engineering & Management, 142(2), 04015066. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set

