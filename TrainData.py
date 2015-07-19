import numpy as np
import cPickle as pickle
import random as  rnd


def loadData(total_files) :
	
	X,y = pickle.load(open("RUNDATA/data_x_test_base.p","rb"))
	stack_size = X.shape[0] ;
	for i in range(total_files):
		print i
		filename = 'RUNDATA/data_x_test_{0}.p'.format(i) ;
		X_,y_ = pickle.load(open(filename,"rb"))
		X = np.append(X,X_,axis=0);
		y= np.append(y,y_);
		stack_size += X_.shape[0] ;

	#RANDOMIZE DATA
	
	rand_index = np.arange(X.shape[0]) ;
	rnd.shuffle(rand_index) ;
	X = X[rand_index,:] ;
	y = y[rand_index] ;
	
	y = np.int_(y);
	return X,y

def loadData_temp() :
	
	X,y = pickle.load(open("RUNDATA/13_22_NEW/data_x_test_12.p","rb"))
	stack_size = X.shape[0] ;
	for i in range(10):
		print i
		filename = 'RUNDATA/13_22_NEW/data_x_test_{0}.p'.format(i+13) ;
		X_,y_ = pickle.load(open(filename,"rb"))
		X = np.append(X,X_,axis=0);
		y= np.append(y,y_);
		stack_size += X_.shape[0] ;

	#RANDOMIZE DATA
	
	rand_index = np.arange(X.shape[0]) ;
	rnd.shuffle(rand_index) ;
	X = X[rand_index,:] ;
	y = y[rand_index] ;
	
	y = np.int_(y);
	return X,y
	
def save_train_and_valid_data(X,y):
	#NORMALIZE X
	#X -= np.mean(X,axis=0);
	#X /= np.std(X,axis=0);
	#
	

 
# divide data into train and validation
	index = 0.8*X.shape[0] ;
	Xtrain = X[0:index,:] ;
	ytrain = y[0:index]
	Xvalid = X[index+1:,:] ;
	yvalid = y[index+1:] ;



	pickle.dump(((Xtrain,ytrain),(Xvalid,yvalid)),open("compiled_unNormalized.p","wb"))
	print 'Combined training and validation data saved as compiled_unNormalized.p'

X,y = loadData(12);
X_9,y_9 = loadData_temp();
X_9 = X_9[:,0:-1] 
#X = X_9 ; y = y_9 ;
#X = np.append(X,X_9,axis=0);
#y= np.append(y,y_9);
save_train_and_valid_data(X,y);