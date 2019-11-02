#  Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
import random
import numpy as np
import matplotlib.pyplot as plt
import mnist
plt.style.use("ggplot")  # Use GGPlot style for graph

def sqDistance(p, q, pSOS, qSOS):
    #  Efficiently compute squared euclidean distances between sets of vectors

    #  Compute the squared Euclidean distances between every d-dimensional point
    #  in p to every d-dimensional point in q. Both p and q are
    #  npoints-by-ndimensions. 
    #  d(i, j) = sum((p(i, :) - q(j, :)).^2)

    d = np.add(pSOS, qSOS.T) - 2*np.dot(p, q.T)
    return d

np.random.seed(1)

#  Set training & testing 
Xtrain, ytrain, Xtest, ytest = mnist.load_data()

#train_size = 10000
test_size  = 10000

train_sample_size = [100, 1000, 2500, 5000, 7500, 10000]
n_folds = [3, 10, 50, 100, 1000]
error_rate_sample = []
error_rate_n_fold = []
error_n_fold_means = []

Xtest = Xtest[0:test_size]
ytest = ytest[0:test_size]

#  Precompute sum of squares term for speed
#XtrainSOS = np.sum(Xtrain**2, axis=1, keepdims=True)

def calc_new_train_SOS(Xtrain_sample):
  XtrainSOS = np.sum(Xtrain_sample**2, axis=1, keepdims=True)
  return XtrainSOS

def calc_new_test_SOS(Xtest_sample):
  XtestSOS  = np.sum(Xtest_sample**2, axis=1, keepdims=True)
  return XtestSOS

#  fully solution takes too much memory so we will classify in batches
#  nbatches must be an even divisor of test_size, increase if you run out of memory

def calc_batch_size(test_size): 
  if test_size > 1000:
    nbatches = 50
  else:
    nbatches = 5
  return nbatches

#  Classify
def classify(Xtest_sample,Xtrain_sample,XtrainSOS_sample,ytrain_sample,XtestSOS_sample,ytest_sample):
  nbatches = calc_batch_size(Xtest_sample.shape[0])
  batches = np.array_split(np.arange(Xtest_sample.shape[0]), nbatches)
  ypred = np.zeros_like(ytest_sample)
  for i in range(nbatches):
    dst = sqDistance(Xtest_sample[batches[i]], Xtrain_sample, XtestSOS_sample[batches[i]], XtrainSOS_sample)
    closest = np.argmin(dst, axis=1)
    ypred[batches[i]] = ytrain_sample[closest]
  return ypred

#  Report
#errorRate = (ypred != ytest).mean()
#print('Error Rate: {:.2f}%\n'.format(100*errorRate))

#  image plot
#plt.imshow(Xtrain[0].reshape(28, 28), cmap='gray')
#plt.show()


# Q1:  Plot a figure where the x-asix is number of training
#      examples (e.g. 100, 1000, 2500, 5000, 7500, 10000), and the y-axis is test error.

# TODO
def predict(Xtest_sample,Xtrain_sample,Xtrain_sample_SOS,ytrain_sample,ytest_sample,XtestSOS):
  prediction = classify(Xtest_sample,Xtrain_sample,Xtrain_sample_SOS,ytrain_sample,XtestSOS,ytest_sample)
  error_rate = (prediction != ytest_sample).mean()
  return error_rate

def n_fold_validation(Xtrain,ytrain,n):
  Xtrain_folds = np.array_split(Xtrain,n)
  ytrain_folds = np.array_split(ytrain,n)
  for i in range(n):
    Xtest_val = Xtrain_folds[i].copy()
    ytest_val = ytrain_folds[i].copy()
    Xtrain_folds_dummy = Xtrain_folds.copy()
    ytrain_folds_dummy = ytrain_folds.copy()
    Xtrain_folds_dummy.pop(i)
    ytrain_folds_dummy.pop(i)
    Xtrain_val = Xtrain_folds_dummy
    ytrain_val = ytrain_folds_dummy
    Xtrain_val_new = np.concatenate(Xtrain_val,axis = 0)
    ytrain_val_new = np.concatenate(ytrain_val,axis = 0)
    Xtrain_val_SOS = calc_new_train_SOS(Xtrain_val_new)
    Xtest_val_SOS = calc_new_test_SOS(Xtest_val)
    error_rate = predict(Xtest_val,Xtrain_val_new,Xtrain_val_SOS,ytrain_val_new,ytest_val,Xtest_val_SOS)
    error_rate_n_fold.append(error_rate)
    error_mean = sum(error_rate_n_fold)/len(error_rate_n_fold)*100  
  error_n_fold_means.append(error_mean)
  return error_n_fold_means
# Q2:  plot the n-fold cross validation error for the first 1000 training training examples

# TODO

def main():
  for size in train_sample_size:
    Xtrain_sample = Xtrain[0:size]
    ytrain_sample = ytrain[0:size]
    Xtrain_sample_SOS = calc_new_train_SOS(Xtrain_sample)
    Xtest_sample_SOS = calc_new_test_SOS(Xtest)
    error_rate = predict(Xtest,Xtrain_sample,Xtrain_sample_SOS,ytrain_sample,ytest,Xtest_sample_SOS)
    error_rate_sample.append(error_rate*100)
  print(error_rate_sample)
  plt.figure()
  plt.plot(train_sample_size,error_rate_sample,marker='o')
  plt.title('Train Set Size vs Error Rate')
  plt.ylabel('Error Rate')
  plt.xlabel('Train Set Size')
  plt.show()

  for fold in n_folds:
    Xtrain_val = Xtrain[0:1000]
    ytrain_val = ytrain[0:1000]
    error_n_fold_means = n_fold_validation(Xtrain_val, ytrain_val, fold)
  print(error_n_fold_means)
  plt.plot(n_folds,error_n_fold_means, marker='o')
  plt.title('n_folds vs Error Rate')
  plt.ylabel('Error Rate')
  plt.xlabel('n_folds')
  plt.show()

if __name__ == '__main__':
    main()