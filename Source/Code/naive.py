#The import dimension (or metric) receives this additional data you are uploading
from sklearn import datasets,metrics
#Support vector machine are set of supervised learning methods used for classification,
#,regression and outliers detection
from sklearn import svm
#we can use naives by using this
from sklearn.naive_bayes import GaussianNB
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

#loading the  dataset
irisdatasets=datasets.load_iris()
#loading  the iris-datasets column values
#getting the data and response of the dataset.
x=irisdatasets.data
y=irisdatasets.target

#split the data of arrays or matrices for training and testing cross validation
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#define the model
model=GaussianNB()
#fit the training data into model
model.fit(x_train,y_train)
#prints the probability of training data
#print(model.score(x_train,y_train))
#define to predict the test data
y_pred = model.predict(x_test)
#calculating the accuracy classification score.

print("The Accuracy classification score of testing : ",metrics.accuracy_score(y_test, y_pred))
