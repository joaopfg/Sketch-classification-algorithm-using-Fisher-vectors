import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

for i in range(8):
	for j in range(80):
		colnames = []

		for k in range(128):
			key = 'V' + str(k+1)
			colnames.append(key)

		file_name = 'Descriptors/descriptors'
		file_name += str(80*i + j)
		file_name += '.csv'

		try:
			df = pd.read_csv(file_name)
		except FileNotFoundError:
			print('File does not exist')
		else:
			df = pd.read_csv(file_name, names=colnames, header=None)
			scaler = MinMaxScaler()
			scaled_data = scaler.fit_transform(df)
			num_inputs = 128 
			num_hidden = 80   
			num_outputs = num_inputs 
			learning_rate = 0.01
			X = tf.placeholder(tf.float32, shape=[None, num_inputs])
			hidden = fully_connected(X, num_hidden, activation_fn=None)
			outputs = fully_connected(hidden, num_outputs, activation_fn=None)
			loss = tf.reduce_mean(tf.square(outputs - X))
			optimizer = tf.train.AdamOptimizer(learning_rate)
			train  = optimizer.minimize( loss)
			init = tf.global_variables_initializer()
			num_steps = 1000

			with tf.Session() as sess:
				sess.run(init)
	
				for iteration in range(num_steps):
					sess.run(train,feed_dict={X: scaled_data})

			with tf.Session() as sess:
				sess.run(init)
		
				output_80d = hidden.eval(feed_dict={X: scaled_data})

			colnames.clear()

			for k in range(80):
				key = 'V' + str(k+1)
				colnames.append(key)

			df = pd.DataFrame(data=output_80d[0:,0:], columns=colnames)

			file_name = 'Descriptors_compressed/descriptors_reduced'
			file_name += str(80*i + j)
			file_name += '.csv'
			df.to_csv(file_name, index=False)

