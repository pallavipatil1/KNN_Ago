from pandas import read_csv
A = read_csv("G:\Python_csv\iris.csv")
Y = A[["Species"]]
X = A.drop(labels=["Unnamed: 0","Species"],axis=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=30)
for i in range(2,20,1):
   from sklearn.neighbors import KNeighborsClassifier
   knc = KNeighborsClassifier(n_neighbors=i)
   model = knc.fit(xtrain,ytrain)
   pred = model.predict(xtest)
   from sklearn.metrics import accuracy_score
   print(i,accuracy_score(ytest,pred))
pred
