## A exercise in risk estimation for FX market making. 

The idea is to try to backtest an estimator on EURCHF 2015 event.

The notebooks contain some experiments (moved near the bottom), done during code development;
but most importantly, notebooks iterate the tasks needed to obtain the estimator.
They call the code placed in 'scripts' dir.

We split the dataset: train on 2009-2013, validate on 2014, and test on 2015 data.

We train the model (XGBoost regressor) for prediction of 5-minutes future realized volatility.

This is the first, baseline version of the estimator.

