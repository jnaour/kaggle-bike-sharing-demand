kaggle-bike-sharing-demand
==========================

Kaggle challenge

#########
Logs:

1st: 
- Model: Random Forest Regressor
- some work on datetime = month + hour
- score: 0.50630

2nd:
- Model: SVR
- Just a quick test with SVR, no real work on model
- score: 1.46266

3rd:
- Model: Random Forest Regressor
- change in criterion (mse in place of Gini)
- score: 0.51055

4th:
- Model: Random Forest Regressor
- change of trees number 500 in places of 100
- score: 0.50714

5th:
- Model: Random Forest Regressor
- some work on outliers to have a more robust model.
drop top 5% worst estimation and rebuild forest
- score: 0.52863

6th:
- Models: RFR
- One model for casual the other for registered
- score: 0.50912

7th:
- Model: RFR
- Better month feature
- score: 0.49587

#########
Ideas: 
- one model for casual and an other for registered
- some work on humidity and windspeed
- exclude outliers for more robustness: didn't improve apparently
- have better month build so that the day that have to be predict are in the middle of the month

