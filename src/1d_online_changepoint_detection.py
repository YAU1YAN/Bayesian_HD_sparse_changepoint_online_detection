from __future__ import division

import matplotlib
matplotlib.use('TkAgg')


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import log, pow

import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.cm as cm
from math import log, pow, sqrt
import os
import csv

column_padding = 20

def calc_d_mean(mean):
	return 1 / (mean[-1] + mean[0])

def check_file_exists(file_name):
	return os.path.isfile(file_name)

def create_title_row_for_raw():
	return ['m1','m2','o1','o2','t1','l_gap','m_delta','o_delta','kl', 'kl2','root_v_a','root_v_delta_z', 'm_a', 'm_z', 'm_delta_z', 'm_delta_t1', 'pre_Ma', 'pre_V', 'tally',\
	'false_tally', '%']

def create_row_for_raw():
	row = {}
	for cell in create_title_row_for_raw():
		row[str(cell)] = 0

	return row

def create_title_row():
	return ['m1','m2','o1','o2','t1','l_gap','m_delta','o_delta','kl', 'kl2', 'A','Z','sum_a','sum_a2','sum_z','sum_z2','m_a', \
	'm_z','v_a','v_z','tally', 'false_tally', 'z_delta', 'sum_delta_z', 'sum_delta_z2', 'm_delta_z', 'v_delta_z', 'delta_t1', \
	'sum_delta_t1', 'sum_delta_t12',  'm_delta_t1', 'v_delta_t1', 'pre_Ma', 'pre_V']

def create_row():
	row = {}
	for cell in create_title_row():
		row[str(cell)] = 0

	return row
def create_key(row):
	return str(row['m1']) + ',' + str(row['m2']) + ',' + str(row['o1']) + ',' + str(row['o2']) + ',' + str(row['t1']) + ',' + str(row['l_gap'])


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

def update_results(mean, varience, lambda_gap, time_series_len, changeDeclare, most_likely, pre_Ma, pre_V, existing_results, existing_results_raw, key):

	if int(changeDeclare) < int(time_series_len):
		print changeDeclare, time_series_len

		existing_results[key]['false_tally'] = int(existing_results[key]['false_tally']) + 1
		existing_results_raw[key]['false_tally'] = existing_results[key]['false_tally']

	else:	
		existing_results[key]['tally'] = int(existing_results[key]['tally']) + 1
		current_tally = int(existing_results[key]['tally'])
		tally_divisor = current_tally - 1 if current_tally > 1 else 1

		m_delta = abs(mean[0] - mean[1])
		o_delta = abs(varience[0] - varience[1])
		kl = log(varience[1] / varience[0]) + (((pow(varience[0], 2)) + pow((mean[0] - mean[1]), 2)) / (2 * pow(varience[1], 2))) - 0.5
		kl2 = log(varience[0] / varience[1]) + (((pow(varience[1], 2)) + pow((mean[1] - mean[0]), 2)) / (2 * pow(varience[0], 2))) - 0.5
		sum_a = float(existing_results[key]['sum_a']) + changeDeclare
		sum_a2 = float(existing_results[key]['sum_a2']) + pow(changeDeclare, 2)
		sum_z = float(existing_results[key]['sum_z']) + most_likely
		sum_z2 = float(existing_results[key]['sum_z2']) + pow(most_likely, 2)

		z_delta = most_likely - time_series_len
		sum_delta_z = float(existing_results[key]['sum_delta_z']) + z_delta
		sum_delta_z2 = float(existing_results[key]['sum_delta_z2']) + pow(z_delta, 2)
		m_delta_z = calculate_mu(sum_delta_z, current_tally)
		v_delta_z = calculate_v(sum_delta_z2, m_delta_z, current_tally, tally_divisor)

		delta_t1 = changeDeclare - time_series_len
		sum_delta_t1 = float(existing_results[key]['sum_delta_t1']) + delta_t1
		sum_delta_t12 = float(existing_results[key]['sum_delta_t12']) + pow(delta_t1, 2)
		m_delta_t1 = calculate_mu(sum_delta_t1, current_tally)
		v_delta_t1 = calculate_v(sum_delta_t12, m_delta_t1, current_tally, tally_divisor)

		m_a = calculate_mu(sum_a, current_tally)
		m_z = calculate_mu(sum_z, current_tally)

		v_a = calculate_v(sum_a2, m_a, current_tally, tally_divisor)
		v_z = calculate_v(sum_z2, m_z, current_tally, tally_divisor)

		existing_results[key]['m1'] = mean[0]
		existing_results[key]['m2'] = mean[1]
		existing_results[key]['o1'] = varience[0]
		existing_results[key]['o2'] = varience[1]
		existing_results[key]['t1'] = time_series_len
		existing_results[key]['l_gap'] = lambda_gap
		existing_results[key]['m_delta'] = m_delta
		existing_results[key]['o_delta'] = o_delta
		existing_results[key]['kl'] = kl
		existing_results[key]['kl2'] = kl2
		existing_results[key]['A'] = changeDeclare
		existing_results[key]['Z'] = most_likely
		existing_results[key]['sum_a'] = sum_a
		existing_results[key]['sum_a2'] = sum_a2
		existing_results[key]['sum_z'] = sum_z
		existing_results[key]['sum_z2'] = sum_z2
		existing_results[key]['m_a'] = m_a
		existing_results[key]['m_z'] = m_z
		existing_results[key]['v_a'] = v_a
		existing_results[key]['v_z'] = v_z

		existing_results[key]['z_delta'] = z_delta
		existing_results[key]['sum_delta_z'] = sum_delta_z
		existing_results[key]['sum_delta_z2'] = sum_delta_z2
		existing_results[key]['m_delta_z'] = m_delta_z
		existing_results[key]['v_delta_z'] = v_delta_z
		existing_results[key]['delta_t1'] = delta_t1
		existing_results[key]['sum_delta_t1'] = sum_delta_t1
		existing_results[key]['sum_delta_t12'] = sum_delta_t12
		existing_results[key]['m_delta_t1'] = m_delta_t1
		existing_results[key]['v_delta_t1'] = v_delta_t1

		existing_results[key]['pre_Ma'] = pre_Ma
		existing_results[key]['pre_V'] = pre_V

		existing_results_raw[key]['m1'] = mean[0]
		existing_results_raw[key]['m2'] = mean[1]
		existing_results_raw[key]['o1'] = varience[0]
		existing_results_raw[key]['o2'] = varience[1]
		existing_results_raw[key]['t1'] = time_series_len
		existing_results_raw[key]['l_gap'] = lambda_gap
		existing_results_raw[key]['m_delta'] = m_delta
		existing_results_raw[key]['o_delta'] = o_delta
		existing_results_raw[key]['kl'] = kl
		existing_results_raw[key]['kl2'] = kl2
		existing_results_raw[key]['root_v_a'] = sqrt(v_a)
		existing_results_raw[key]['root_v_delta_z'] = sqrt(v_delta_z)
		existing_results_raw[key]['m_a'] = m_a
		existing_results_raw[key]['m_z'] = m_z
		existing_results_raw[key]['m_delta_z'] = m_delta_z
		existing_results_raw[key]['m_delta_t1'] = m_delta_t1
		existing_results_raw[key]['tally'] = current_tally

		existing_results_raw[key]['pre_Ma'] = pre_Ma
		existing_results_raw[key]['pre_V'] = pre_V

	current_tally = int(existing_results[key]['tally'])
	tally_divisor = current_tally - 1 if current_tally > 1 else 1

	existing_results_raw[key]['%'] = int(existing_results[key]['false_tally']) / tally_divisor

	return existing_results, existing_results_raw

def new_result(mean, varience, lambda_gap, second_time_series_len, changeDeclare, most_likely, pre_Ma, pre_V, existing_results, existing_results_raw, key):
	existing_results[key] = create_row()
	existing_results_raw[key] = create_row_for_raw()
	return update_results(mean, varience, lambda_gap, second_time_series_len, changeDeclare, most_likely, pre_Ma, pre_V, existing_results, existing_results_raw, key)


def log_result(mean, varience, lambda_gap, second_time_series_len, changeDeclare, most_likely, pre_Ma, pre_V):

	key = create_key({'m1': mean[0], 'm2': mean[1], 'o1': varience[0], 'o2': varience[1], 't1': second_time_series_len, 'l_gap': lambda_gap})

	existing_results = get_existing_results('rolling_results_1d.csv')
	existing_results_raw = get_existing_results('raw_results_1d.csv')

	results = {}
	if key in existing_results:
		results, raw_results = update_results(mean, varience, lambda_gap, second_time_series_len, changeDeclare, most_likely, pre_Ma, pre_V, existing_results, existing_results_raw, key)
	else:
		results, raw_results = new_result(mean, varience, lambda_gap, second_time_series_len, changeDeclare, most_likely, pre_Ma, pre_V, existing_results, existing_results_raw, key)

	write_results(results, 'rolling_results_1d.csv', create_title_row())
	write_results(raw_results, 'raw_results_1d.csv', create_title_row_for_raw())

def  calculate_mu(val, tally):
	return (val / tally)

def calculate_v(sum_val2, mu, tally, tally_divisor):
	return ((sum_val2 / tally) - pow(mu, 2)) * (tally / tally_divisor)
	

def generate_normal_time_series(num, mean, varience, first_time_series_len, second_time_series_len):
	data = np.array([], dtype=np.float64)
	partition = [first_time_series_len, second_time_series_len]

	if len(partition) != len(mean) and len(partition) != len(varience):
		print "array len mismatch"
		return

	for i in range (0, len(partition)):
		if varience[i] < 0:
			varience[i] = varience[i] * -1

		tdata = np.random.normal(mean[i], varience[i], partition[i])
		data = np.concatenate((data, tdata))

	return data

def three_plots(data, R, first_time_series_len,second_time_series_len, nw=1):
	fig, ax = plt.subplots(figsize=[18, 16])
	ax = fig.add_subplot(3, 1, 1)
	ax.plot(data)
	ax = fig.add_subplot(3, 1, 2, sharex=ax)
	sparsity = 5  # only plot every fifth data for faster display
	ax.pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
			  np.array(range(0, len(R[:,0]), sparsity)), 
			  -np.log(R[2:-1:sparsity, 2:-1:sparsity]), 
			  cmap=cm.Greys, vmin=0, vmax=30)
	ax = fig.add_subplot(3, 1, 3, sharex=ax)
#################################################################
	#thing = R[:,nw+first_time_series_len] ##Use changeDeclare[0] for this instead of nw
	#thing = np.insert(thing,0,np.zeros(first_time_series_len))
	#thing = list(thing)
	#thing = thing[:first_time_series_len+second_time_series_len]
	#ax.plot(thing)
#########################################################################
	ax.plot((np.insert((R[nw,nw+2:-1]),0,0)/np.sum(R[nw,nw+1:-1])))	 #this plot where the programe thinks the changed point is located
												   #, aka, the PDF of the changepoint, after nw delay, and normalized 
												   #ignore the 1st point, where a changed has occured at default
	plt.show()
def create_plot(title, i):
	fig = plt.figure(figsize=[18, 16])
	
	ax = fig.add_subplot(1, 1, 1)

	plt.title(title)

	return fig, ax

def plot_graph(fig, ax, data, results, clear=True, nw=1):
	if clear:
		ax.clear()


	ax.plot(results[0][nw,nw:-1], label=str(results[1])) # Plot a single dim array, from row N, position M to end (-1)
						  	 							 # i.e. row 1, from position 1 to end
	legend = ax.legend(loc='upper right', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	for label in legend.get_texts():
		label.set_fontsize('large')
	for label in legend.get_lines():
		label.set_linewidth(1.5)

	fig.canvas.draw()




def calculate_with_mean_change():
	file_name = "1d_online_changepoint_results_mean.out"
	clear_log(file_name)

	num = 2
	mean = [0, [1, 1.1, 1.5, 1.7, 2, 10]]
	varience = [1, 1]
	nw = 20

	alpha = 0.1
	beta = 0.01
	M_ = 0

	first_time_series_len = 500
	second_time_series_len = 500
	lambda_gap = 500

	fig, ax = create_plot("Mean Change", 1)

	results = []

	for m in mean[1]:
		data = generate_normal_time_series(num, [mean[0], m], varience, first_time_series_len, second_time_series_len)

		if data is None:
			quit()

		R, maxes = online_changepoint_det(data, lambda_gap, alpha, beta, M_)

		results.append([R, m])

		for i, el in reversed(list(enumerate(results))):

			plot_graph(fig, ax, data, el, (i == (len(results) - 1)), nw)

			k = raw_input('Press Enter to add next illustration...')





def online_changepoint_det(data, lamda_gap, alpha, beta, M_):
	return oncd.online_changepoint_detection(data, partial(oncd.constant_hazard, lamda_gap), oncd.StudentT(alpha, beta, 1, M_))

def calculate_with_3_charts():
	mean = [0, 10]
	varience = [7.5, 7.5]
	first_time_series_len =500
	lambda_gap = 1000
	print "(1)mu1= ", mean[0]
	print "(2)mu2= "
######################################################################
	second_time_series_len = 1200
	num = 2
	data = generate_normal_time_series(num, mean, varience, first_time_series_len, second_time_series_len)

	if data is None:
		quit()

	alpha = 0.1
	beta = 0.01
	M_ = 0
#
#	pre_Ma = M_
#	pre_V = (beta + 1) / alpha

	R, maxes, mu= online_changepoint_det(data, lambda_gap, alpha, beta, M_)

	i=0
	growth = []
	changeDeclare = []
	while i < first_time_series_len + second_time_series_len:
		growth.append(1.0-R[i,i])  #the ith row plus and ith coloum(diagonal) is the probability of no change ever happened
		#print sum(R[:,i])	 #this is to prove that all colom are correctly normalized
		if 1.0-R[i,i] > 0.9 and 1.0-R[i,i] <1.0 and len(changeDeclare)==0:
			changeDeclare.append(i)
		i = i+1
	if len(changeDeclare)==0:
		changeDeclare.append(i)
		print "no change detected"
	nw = changeDeclare[0] - first_time_series_len 
		#################################################################
 	#fig, ax = create_plot("tracker", 0)
	
	#ax = fig.add_subplot(3, 1, 1)
 #	red_patch = mpatches.Patch(color='red', label='mu')
 #	plt.legend(handles=[red_patch])
	#ax.plot(maxes, label = "most likely path R max")   #plot naive most likely path
	#ax = fig.add_subplot(3, 1, 2)
	#red_patch = mpatches.Patch(color='red', label='variance')
 #	plt.legend(handles=[red_patch])
	#ax.plot(mu, label = "m")
	#ax = fig.add_subplot(3,1,3)
	#red_patch = mpatches.Patch(color='red', label='alarm'+str(8))
 #	plt.legend(handles=[red_patch])
	#ax.plot(growth, label = "chance of 1 change has already occured")  #plot change of a singular change point has already happened
	#fig.canvas.draw()
	####################################################################
	#three_plots(data, R, first_time_series_len,second_time_series_len, nw)
	####################################################################
	print "all need to log parameters: ", mean[0], mean[1], varience[0], varience[1], first_time_series_len
	print "change declare at:" ,changeDeclare	  
	print "change point location(most likely):  ", 2 + R[nw,nw+2:-1].argmax()  ##find the mode of the PDF after nw delay, ignore the first 2 points using row
 #	   print "change point location(also likely):  ", first_time_series_len + R[:,changeDeclare[0]].argmax() - nw #this uses 
	#
	print "KL: ", log(varience[1] / varience[0]) + (((pow(varience[0], 2)) + pow((mean[0] - mean[1]), 2)) / (2 * pow(varience[1], 2))) - 0.5
	print "KL (2): ", log(varience[0] / varience[1]) + (((pow(varience[1], 2)) + pow((mean[1] - mean[0]), 2)) / (2 * pow(varience[0], 2))) - 0.5 ##this seem to be the correct one, the earlier time series variance at the top
	print "Lamda_gap", lambda_gap


	kl = log(varience[1] / varience[0]) + (((pow(varience[0], 2)) + pow((mean[0] - mean[1]), 2)) / (2 * pow(varience[1], 2))) - 0.5
	kl2 = log(varience[0] / varience[1]) + (((pow(varience[1], 2)) + pow((mean[1] - mean[0]), 2)) / (2 * pow(varience[0], 2))) - 0.5



	pre_Ma = M_
	pre_V = (beta + 1) / alpha

	log_result(mean, varience, lambda_gap, first_time_series_len, changeDeclare[0], (2 + R[nw,nw+2:-1].argmax()), pre_Ma, pre_V)




####################################################


#plt.ion()

#calculate_with_lambda_change()
#calculate_with_mean_change()
#calculate_with_nw_change()
for i in range (0,20):
    calculate_with_3_charts()

k = raw_input('Press Enter to close... ')
