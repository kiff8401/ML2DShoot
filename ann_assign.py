from __future__ import division
import  numpy as np
#import matplotlib.pyplot as pl
import cPickle as pickle
import gzip 




class MLP(object):
	def __init__(self,nin,nout,nhidden):
		rng = np.random.RandomState(0)
		self.W_hidden = rng.uniform(size= (nin,nhidden),low = -np.sqrt(6/(nin + nhidden)),
		high = np.sqrt(6/ (nin+nhidden))) ;
		
		self.b_hidden = np.zeros(nhidden);
		self.W_out = np.zeros((nhidden,nout)) ;
		self.b_out = np.zeros(nout);
		self.params = (self.W_hidden,self.b_hidden,self.W_out,self.b_out);

	def forward(self, X):
		"""
		Given inputs 'X' as an (n,d)-array compute and return the hidden unit
		activations and the output unit activations. Return this as a tuple
		(z,p) where these are arrays of size (n,nhidden) and (n,nout)
		respectively.
		"""
		# z contains the activations of the hidden units
		z = np.tanh(np.dot(X,self.W_hidden) + self.b_hidden) ;
		a = np.dot(z,self.W_out) + self.b_out ;
		a -= np.log(np.sum(np.exp(a),axis=1)).reshape(-1,1)
		p = np.exp(a,out=a);
		return z,p
	
	def output_single_Vector(self,X):
		z = np.tanh(np.dot(X,self.W_hidden) + self.b_hidden) ;
		a = np.dot(z,self.W_out) + self.b_out ;
		a -= np.log(np.sum(np.exp(a)))
		return np.exp(a,out=a);

	def log_likelihood(self,X,y):
		z,p = self.forward(X);
		i = np.arange(0,p.size,p.shape[1]) + y ;
		return np.mean(np.log(p.flat[i]));
		
	def error(self,X,y):
		z,p = self.forward(X);
		return np.mean(y!= np.argmax(p,axis=1)) ;
		
	def gradient(self,X,y):
                #(dW_hidden, db_hidden, dW_out, db_out)
                z,p = self.forward(X);
                n,nout = p.shape ;
                # db_out 	
                T  = np.zeros((n,nout));
                i = np.arange(0,p.size,p.shape[1]) + y ;	 
                T.flat[i]  = 1.0 ;
                db_out =np.mean(p-T,axis=0);
                #dW_out
                dW_out = (np.dot(np.transpose(z),(p-T)))/n;
                #db_hidden   .
                W_out_ = self.W_out[:,y] - np.dot(self.W_out,np.transpose(p));
                z_2 = 1.0-np.transpose(z)*np.transpose(z);
                db_hidden = -np.mean(W_out_*z_2,axis=1);
                #dW_hidden
                temp = W_out_*z_2 ;
                dW_hidden = -np.dot(np.transpose(X),np.transpose(temp))/n ;
                return dW_hidden, db_hidden, dW_out, db_out
		
def train_model(model,data,alpha,batchsize,nepochs):
	(Xtrain,ytrain),(Xvalid,yvalid) = data ;
	ntrain = int(Xtrain.shape[0]/batchsize);
	nvalid = int(Xvalid.shape[0]/batchsize);
	b = lambda i: slice(i*batchsize, (i+1)*batchsize) ;
	for epoch in xrange(nepochs):
		for i in xrange(ntrain):
			for (p,g) in zip(model.params, model.gradient(Xtrain[b(i)], ytrain[b(i)])):
				#print p.shape;
				p -= alpha * g
		error = np.mean([model.error(Xvalid[b(i)], yvalid[b(i)]) for i in xrange(nvalid)]) ;
		print 'epoch %2d, validation error: %.4f' % \
			(epoch,error)
		

	

if __name__ == '__main__':
	print '... loading data'
	data = pickle.load(open("compiled_unNormalized.p","rb"))
	(Xtrain,ytrain), (Xvalid,yvalid) = data
	print Xtrain.shape
	print ytrain.shape
	print '... building/training the model'
	
	model = MLP(nin=8, nout=4, nhidden=10)
	train_model(model, data, alpha=0.01, batchsize=20, nepochs=110)
	"""
	# Testing the model for all missclassified 4s
	allfours = Xvalid[yvalid==4,:];
        y_4s = yvalid[yvalid==4] ;
        z,p_fours = model.forward(allfours);
        allfours_miss = allfours[y_4s!= np.argmax(p_fours,axis=1),:];
        for i in xrange(0,20):
                pl.subplot(5,4,i);
                pl.imshow(allfours_miss[i,:].reshape(28,28), cmap='gray')

        pl.show();
	"""
	
	
	
	
	
