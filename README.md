## A exercise in risk estimation for FX market making. 

The idea is to try to backtest an estimator on EURCHF 2015 event.

First, please have the tick data fetched from OANDA servers
(this was being experimented on in the 'notebooks/oanda_data_fetcher.ipynb'),
see the 'scripts/fetch_data.py'. 

Feature engineering is done in 'scripts/build_features.py'.

We split the dataset: train on 2009-2013, validate on 2014, test on 2015; see 'scripts/split_dataset.py'. 
The data, fetched with this code, required some cleaning - implemented in 'risk_estimator/data_loader.py'.

We train the model for prediction of 5min future realized volatility in 'scripts/train_models.py',
and test it in 'notebooks/vol_model.ipynb', with the calibration curve plotted using 'risk_estimator/plotting.py'.

This is the first, baseline version of the estimator.

