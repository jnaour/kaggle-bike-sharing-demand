kaggle-bike-sharing-demand
==========================

Kaggle challenge

#########
Logs:

-1st try: 
Model: Random Forest Regressor
some work on datetime = month + hour
score: 0.50630

-2nd try:
Model: SVR
Just a quick test with SVR, no real work on model
score: 1.46266

-3rd try:
Model: Random Forest Regressor
change in criterion (mse in place of Gini)
score: 0.51055

-4th try:
Model: Random Forest Regressor
change of trees number 500 in places of 100
score: 0.50714

-5th try:
Model: Random Forest Regressor
some work on outliers to have a more robust model
drop top 5% worst estimation and rebuild forest
score: 0.52863



#########
Ideas: 
- one model for casual and an other for registered
- some work on humidity and windspeed
- exclude outliers for more robustness: didn't work apparently
- have better month build so that the day that have to be predict are in the middle of the month

