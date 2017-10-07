Bayesian online changepoint detection for high dimensional sparse changepoint

Assuming signal being generated in a time series following a particular probability distribution and is expected to experience an
 arupt change in parameter that defines the probability distribution. Detecting the location of the changepoint inside the timeseries known as the change point detection problem. This project extend the changepoint detection algorithm into an online/sequential algorithm. Furthermore, I extended it to detect changes in high dimensional signal, where only a sparse subset of the dimensions are experiencing the arupt change, and that the change in each of the dimension is too small to be detected individually by a 1D changepoint detection algorithm, therefore a sort online PCA is used borrowed strength across the dimensions to make the detection algorithm more sensitive to these smaller changes.

How To Run

Run "multiD_online_changepoint_detection.py", use the high_dimensional_test() to control the nature of the timeseries and the initializing paraments of the detection algorithm. The results, input parameter, and the performace of the detection algorithm would be automatically logged at rolling_results_3d.csv  in a form of a spread sheet

