from flask import Flask, render_template, request

import tensorflow as tf 
import numpy as np 
import datetime 
app = Flask(__name__)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([4, 1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
model = tf.compat.v1.global_variables_initializer() 


sess = tf.compat.v1.Session()



saver = tf.compat.v1.train.Saver
saver = tf.compat.v1.train.import_meta_graph('./model/rice.cpkt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))
@app.route("/", methods = ['GET', 'POST'])

def index():



	if request.method == 'GET' :
		return render_template('index.html') 

	if request.method == 'POST' :


		avg_temp = float(request.form['avg_temp'])
		min_temp = float(request.form['min_temp'])
		max_temp = float(request.form['max_temp'])
		rain_fall = float(request.form['rain_fall'])


	
		
		sess.run(model)
	
	
		data = ((avg_temp, min_temp, max_temp, rain_fall),(0,0,0,0))
		arr = np.array(data, dtype=np.float32)
		x_data = arr[0:4]
		dict = sess.run(hypothesis, feed_dict={X: x_data})

		price = dict[0] 	
		return render_template("index.html" , price = price*100) 

if __name__ == '__main__':
	app.run(debug = True) 



