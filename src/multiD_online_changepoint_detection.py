from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn

import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.cm as cm
from math import log, pow, sqrt
import os
import csv

def calc_d_mean(mean):
	return 1 / (mean[-1] + mean[0])

def check_file_exists(file_name):
	return os.path.isfile(file_name)

def create_title_row_for_raw():
	return ['o', 't', 'k', 'k_hat', 'sum_k_hat', 'sum_k_hat2', 'm_k_hat', 'v_k_hat', 'percent', 'percent2', 'sum_percent', 'sum_percent2', 'm_percent', 'v_percent',\
	'pre_Ma', 'pre_V', 'delta_t', 'delta_tb', 'sum_delta_t', 'sum_delta_t2', 'm_delta_t', 'v_delta_t', 'tally', 'false_tally', '%']

def create_row_for_raw():
	row = {}
	for cell in create_title_row_for_raw():
		row[str(cell)] = 0

	return row

def create_title_row():
	return ['o', 't', 'k', 'k_hat', 'percent', 'm_k_hat', 'm_percent', 'v_percent', 'delta_t', 'delta_tb', 'm_delta_t', 'v_delta_t', \
	'pre_Ma', 'pre_V','tally', 'false_tally']

def create_row():
	row = {}
	for cell in create_title_row():
		row[str(cell)] = 0

	return row

def create_key(row):
	return str(row['k']) + ',' + str(row['t']) + ',' + str(row['o'])


def get_existing_results(name):
	result = {}
	if check_file_exists(name):
		with open(name, 'rb') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				result[create_key(row)] = row

	return result

def write_results(results, name, title):
	with open(name, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=title)
		writer.writeheader()
		for _, value in results.iteritems():
			writer.writerow(value)

def update_results(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V, existing_results, existing_results_raw, key):


	delta_t = np.amin(declare_list)-first_time_series_len

	if int(delta_t) < 0:
		existing_results[key]['false_tally'] = int(existing_results[key]['false_tally']) + 1
		existing_results_raw[key]['false_tally'] = existing_results[key]['false_tally']

	else:	
		existing_results[key]['tally'] = int(existing_results[key]['tally']) + 1
		current_tally = int(existing_results[key]['tally'])
		tally_divisor = current_tally - 1 if current_tally > 1 else 1

		o = sparcity[0]
		k_hat = np.argmin(declare_list)+1
		percent = correctness[np.argmin(declare_list)]
		delta_tb = np.amin(declare)

		delta_t2 = pow(delta_t, 2)
		sum_delta_t = float(existing_results_raw[key]['sum_delta_t']) + delta_t
		sum_delta_t2 = float(existing_results_raw[key]['sum_delta_t2']) + delta_t2
		m_delta_t = calculate_mu(sum_delta_t, current_tally)
		v_delta_t = calculate_v(sum_delta_t2, m_delta_t, current_tally, tally_divisor)


		k_hat2 = pow(k_hat, 2)
		sum_k_hat = float(existing_results_raw[key]['sum_k_hat']) + k_hat


		print k_hat, sum_k_hat


		sum_k_hat2 = float(existing_results_raw[key]['sum_k_hat2']) + k_hat2
		m_k_hat = calculate_mu(sum_k_hat, current_tally)
		v_k_hat = calculate_v(sum_k_hat2, m_k_hat, current_tally, tally_divisor)

		percent2 = pow(percent, 2)
		sum_percent = float(existing_results_raw[key]['sum_percent']) + percent
		sum_percent2 = float(existing_results_raw[key]['sum_percent2']) + percent2
		m_percent = calculate_mu(sum_percent, current_tally)
		v_percent = calculate_v(sum_percent2, m_percent, current_tally, tally_divisor)

		existing_results_raw[key]['o'] = o
		existing_results_raw[key]['t'] = first_time_series_len
		existing_results_raw[key]['k'] = K1
		existing_results_raw[key]['k_hat'] = k_hat

		print k_hat, sum_k_hat


		existing_results_raw[key]['sum_k_hat'] = str(sum_k_hat)
		existing_results_raw[key]['sum_k_hat2'] = sum_k_hat2
		existing_results_raw[key]['m_k_hat'] = m_k_hat
		existing_results_raw[key]['v_k_hat'] = v_k_hat
		existing_results_raw[key]['percent'] = percent
		existing_results_raw[key]['percent2'] = percent2
		existing_results_raw[key]['sum_percent'] = sum_percent
		existing_results_raw[key]['sum_percent2'] = sum_percent2
		existing_results_raw[key]['m_percent'] = m_percent
		existing_results_raw[key]['v_percent'] = v_percent
		existing_results_raw[key]['delta_t'] = delta_t
		existing_results_raw[key]['delta_tb'] = delta_tb
		existing_results_raw[key]['sum_delta_t'] = sum_delta_t
		existing_results_raw[key]['sum_delta_t2'] = sum_delta_t2
		existing_results_raw[key]['m_delta_t'] = m_delta_t
		existing_results_raw[key]['v_delta_t'] = v_delta_t
		existing_results_raw[key]['pre_V'] = pre_V
		existing_results_raw[key]['pre_Ma'] = pre_Ma
		existing_results_raw[key]['tally'] = current_tally

		existing_results[key]['o'] = o
		existing_results[key]['t'] = first_time_series_len
		existing_results[key]['k'] = K1
		existing_results[key]['k_hat'] = k_hat
		existing_results[key]['m_k_hat'] = m_k_hat
		existing_results[key]['percent'] = percent
		existing_results[key]['m_percent'] = m_percent
		existing_results[key]['v_percent'] = v_percent
		existing_results[key]['delta_t'] = delta_t
		existing_results[key]['delta_tb'] = delta_tb
		existing_results[key]['m_delta_t'] = m_delta_t
		existing_results[key]['v_delta_t'] = v_delta_t
		existing_results[key]['pre_V'] = pre_V
		existing_results[key]['pre_Ma'] = pre_Ma

	current_tally = int(existing_results[key]['tally'])
	tally_divisor = current_tally - 1 if current_tally > 1 else 1

	existing_results_raw[key]['%'] = int(existing_results[key]['false_tally']) / tally_divisor

	return existing_results, existing_results_raw

def new_result(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V, existing_results, existing_results_raw, key):
	existing_results[key] = create_row()
	existing_results_raw[key] = create_row_for_raw()
	return update_results(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V, existing_results, existing_results_raw, key)


def log_result(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V):

	key = create_key({'t': first_time_series_len, 'k': K1, 'o': sparcity[0] })

	existing_results = get_existing_results('rolling_results_3d.csv')
	existing_results_raw = get_existing_results('raw_results_3d.csv')

	results = {}
	if key in existing_results:
		results, raw_results = update_results(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V, existing_results, existing_results_raw, key)
	else:
		results, raw_results = new_result(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V, existing_results, existing_results_raw, key)

	write_results(results, 'rolling_results_3d.csv', create_title_row())
	write_results(raw_results, 'raw_results_3d.csv', create_title_row_for_raw())

def  calculate_mu(val, tally):
	return (val / tally)

def calculate_v(sum_val2, mu, tally, tally_divisor):
	return ((sum_val2 / tally) - pow(mu, 2)) * (tally / tally_divisor)


def generate_normal_time_series(mean, sparcity, varience, first_time_series_len, second_time_series_len):
	data = np.array([], dtype=np.float64)
	partition = [first_time_series_len, second_time_series_len]

	for i in range(0, len(partition)):
		if varience[i] < 0:
			varience[i] = varience[i] * -1

		tdata = np.random.normal((mean if i == 0 else (mean + sparcity)), varience[i], partition[i])
		data = np.concatenate((data, tdata))

	return data

def generate_multi_dimensional_series(dimensions=1, sparcity=[1], mean=0, varience=[1, 1], first_time_series_len=100, second_time_series_len=101):
	if dimensions != len(sparcity):
		print "mean, dimensions mismatch"
		return

	data = []

	for i in range(0, dimensions):
		data.append(generate_normal_time_series(mean, sparcity[i], varience, first_time_series_len, second_time_series_len))

	return data

def create_projected_dimension(data, v):
	projected_dimension = np.array([], dtype=np.float64)

	for i in range(0, len(data[0])):
		base_array = np.array([], dtype=np.float64)
		for el in data:
			base_array = np.append(base_array, el[i])

		dimension = np.dot(base_array, v)

		projected_dimension = np.append(projected_dimension, dimension)

	return projected_dimension

def plots(data):
	fig, ax = plt.subplots(figsize=[18, 16])
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(data)

	plt.show()

def create_plot(title, i):
	fig = plt.figure(figsize=[18, 16])
	
	ax = fig.add_subplot(1, 1, 1)

	plt.title(title)

	return fig, ax

def plot_graph(fig, ax, data, clear, dimension):
	#if clear:
	#	ax.clear()

	ax.plot(data, label=dimension) # Plot a single dim array, from row N, position M to end (-1)
								 							 # i.e. row 1, from position 1 to end
	legend = ax.legend(loc='upper right', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	for label in legend.get_texts():
		label.set_fontsize('large')
	for label in legend.get_lines():
		label.set_linewidth(1.5)

	fig.canvas.draw()

def plot_all(dimensions, data, seperation, title):
	fig, ax = create_plot(title, 0)
	for i in range(0, dimensions):
		plot_graph(fig, ax, data[i]+(i*seperation), 1, "Dimension " + str(i+1))

def HDonline_changepoint_det(data, lamda_gap, alpha, beta, M):
	return oncd.HDSonline_changepoint_detection(data, partial(oncd.constant_hazard, lamda_gap), oncd.StudentT(alpha, beta, 1, M))

def multi_dimensional_changepoint(dimensions, data, lamda_gap, alpha, beta, M_):
	X = []
	P = []
	M = []
	HDdeclare = []

	for i in range(0, dimensions):
		R, mean, cProb, declare = HDonline_changepoint_det(data[i], lamda_gap, alpha, beta, M_)
		HDdeclare.append(declare)
		X.append(data[i])
		M.append(mean)
		P.append(cProb)
	

	return X,P,M, HDdeclare
def K_largest_argument(vector, K=1): #find the Kth largest element in 
	vector = np.array(vector)
	index = vector.argsort()[K*-1:]		
	return index
		
def projection(dimensions, P, M, K=1):
	P = np.transpose(P)	 #this is to make sure that each time instance enter as rows, for enumerate to work through time
	v = np.zeros((dimensions, len(P) ))#this has time going horizontally
	M = np.transpose(M)
	strongestIndex = []
	for t,i in enumerate (P):
		 index = K_largest_argument(i,K)
		 strongestIndex.append(index)
		 v[index,t]=1
	
	return v , strongestIndex

def project_on_X(X,v):
	X = np.transpose(X)
	v = np.transpose(v)
	projected_x = np.zeros(len(X))
	for t,x in enumerate(X):
		projected_x[t] = np.dot(x,v[t,:])
	return projected_x
 
def illustrate_K(K,data,dimensions,sparcity,lamda_gap, mean, varience, first_time_series_len, second_time_series_len,P,M,X, alpha, beta, m):
	KL_divergence = []
	for i in range(0,dimensions):
		KL_divergence.append(log(1.0 / 1.0) + (((pow(1.0, 2)) + pow((0.0 - sparcity[i]), 2)) / (2 * pow(1.0, 2))) - 0.5)

	print "KL divergence", np.around(KL_divergence,4)
	
	for i in range(0,dimensions):
		print "D",str(i),"stepsize: ",sparcity[i]

		
		v, strongestIndex = projection(dimensions, P, M, K)
		projected_x = project_on_X(X,v)
		
		R, pro_mean, cProb, pro_declare = HDonline_changepoint_det(projected_x, lamda_gap, alpha, beta, m) #this is a simple 1D on the projected
		
		print "correct strongest index: ", K_largest_argument(sparcity,K)
		s = np.zeros(len(sparcity))
		s[K_largest_argument(sparcity,K)]=1
		print "stongest index at declare: ", strongestIndex[pro_declare]	#at declare, guess K most likely changed dimension
		print "correct projection: ", s
		
	print "projection at declare: ", v[:,pro_declare]
	
        X = np.vstack((X,projected_x))
	plot_all(dimensions+1,X,10, "data+projected data") 
	P = np.vstack((P,cProb))
	plot_all(dimensions+1,P,2, "changepoint probabitly")
	M = np.vstack((M,pro_mean))
	plot_all(dimensions+1,M,10, "mean inference")
	#
	print "change point location(most likely):	", 2 + R[pro_declare-first_time_series_len,pro_declare-first_time_series_len+2:-1].argmax()

	print "pro_declare: ", pro_declare
	
#	mean = 0
#	varience = [1, 1]
#	sparcity = [5, 0, -8]
#	first_time_series_len = 500
#	second_time_series_len = 500+500
#	dimensions = 3
#
#	data = generate_multi_dimensional_series(dimensions, sparcity, mean, varience, first_time_series_len, second_time_series_len)
#
#	v = np.array([1, 0, -1], dtype=np.float64)
#
#	d = create_projected_dimension(data, v)
#
#	fig, ax = create_plot("Test", 0)
#	plt.plot(projected_x+20)
#        plt.plot(d)
#        plt.plot(data[0]-10)
#        plt.plot(data[1]-20)
#        plt.plot(data[2]-30)
#	print "mean_before_p", np.mean(projected_x[:500])
#	print "variance_before_p", np.var(projected_x[:500])
#	print "mean_p", np.mean(projected_x[500:])
#	print "variance_p", np.var(projected_x[500:])
#	print "mean_before", np.mean(d[:500])
#	print "variance_before", np.var(d[:500])
#	print "mean", np.mean(d[500:])
#	print "variance", np.var(d[500:])
	plt.show()

def high_dimensional_test():
	mean = 0
	varience = [1, 1]
	sparcity = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
	first_time_series_len = 500
	second_time_series_len = 500+500
	dimensions = len(sparcity)
	lamda_gap = 1000
	K1=9#this is for the correct sparcity
	data = generate_multi_dimensional_series(dimensions, sparcity, mean, varience, first_time_series_len, second_time_series_len)


	alpha = 0.1
	beta = 0.01
	M_ = 0
	
	X, P , M, declare = multi_dimensional_changepoint(dimensions, data, lamda_gap, alpha, beta, M_)# this is 1D detection on each 
	print "declare stacks", declare	#this just print out without projection, if it reads end points i.e 1000, no change has occured
	s = np.zeros(len(sparcity))	#find the correct projection
	s[K_largest_argument(sparcity,K1)]=1
		
	#illustrate_K(K1,data,dimensions,sparcity,lamda_gap, mean, varience, first_time_series_len, second_time_series_len,P,M,X,alpha, beta, M_)
	
	#######################################
	declare_list = []
	correctness = []

	for i in range (0,dimensions):
		K=i+1
		v, strongestIndex = projection(dimensions, P, M, K)
		projected_x = project_on_X(X,v)
		R, pro_mean, cProb, pro_declare = HDonline_changepoint_det(projected_x, lamda_gap, alpha, beta, M_) #this is a simple 1D on the projected
		declare_list.append(pro_declare)
		print "for K = ", str(K), "declare made at", pro_declare
		print "correct projection", s, "projection @ delcare", v[:,pro_declare]
		print "correctness", ((dimensions-np.sum(np.absolute(s-v[:,pro_declare])))/dimensions)
		correctness.append(((dimensions-np.sum(np.absolute(s-v[:,pro_declare])))/dimensions))
		


	pre_Ma = M_
	pre_V = (beta + 1) / alpha


	print "important parameter to keep track	", first_time_series_len, K1,sparcity[0]
	print "most likely K = ", np.argmin(declare_list)+1, "correctioness: ", correctness[np.argmin(declare_list)], "delay: ", np.amin(declare_list)-first_time_series_len
	print "delay_before: ", np.amin(declare)

	log_result(first_time_series_len, K1, sparcity, declare, declare_list, correctness, pre_Ma, pre_V)

for i in range (0,10):		
    high_dimensional_test()


