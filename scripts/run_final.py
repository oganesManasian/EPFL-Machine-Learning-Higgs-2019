from helpers import *
from implementations import *
from preprocessing_data import *

DATA_TRAIN_PATH = "../data/train.csv"
DATA_TEST_PATH = "../data/test.csv" 

def log_features(data):
    log_tX = data[:,1:].copy()
    for i in range(log_tX.shape[1]):
        val = 0
        mn = np.min(log_tX[:,i])
        if mn<=0:
            val = np.abs(mn) + 0.001
        b = np.log(val + log_tX[:,i])
        log_tX[:,i] = b
    return log_tX

def poly_features(data,degree=2):
    tX_new = data[:,:1].copy()
    for i in range(1,data.shape[1]):
        tX_new = np.column_stack((tX_new,data[:,i]))
        for j in range(degree-1):        
            tX_new = np.column_stack((tX_new,data[:,i]**(j+2)))
    return tX_new


if __name__ == "__main__":

	print('Importing data...')
	y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
	_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

	tX_stacked = np.vstack((tX, tX_test))
	tX_stacked = fill_missing_values(tX_stacked,mute=True)
	tX_stacked = np.column_stack((np.ones(tX_stacked.shape[0]),tX_stacked))

	print('Feature engineering...')
	log_tX_stacked = log_features(tX_stacked)
	tX_stacked_new = poly_features(tX_stacked,degree=6)
	tX_stacked_new = np.column_stack((tX_stacked_new,log_tX_stacked))

	tX_new, tX_test_new = np.split(tX_stacked_new,[tX.shape[0]])

	train_x, train_y, test_x, test_y = split_data(tX_new,y,0.8)

	print('Fitting the model')
	w,loss = ridge_regression(train_y,train_x,lambda_=0) # equivalent to just least squares as we take lambda=0

	prediction_for_train = predict_labels(w,train_x)
	print('Train set accuracy = {}%'.format(compute_accuracy(train_y,prediction_for_train)))

	prediction_for_test = predict_labels(w,test_x)
	print('Test set accuracy = {}%'.format(compute_accuracy(test_y,prediction_for_test)))

	print('Fitting the model on the whole data')
	w,_ = ridge_regression(y,tX_new,lambda_=0) 

	print('Creating submission for the real test data')
	prediction = predict_labels(w,tX_test_new)
	create_csv_submission(ids_test,prediction,'submission')
	print('Done!')


