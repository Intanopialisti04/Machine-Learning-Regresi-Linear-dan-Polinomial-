from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
#import warnings
from warnings import simplefilter
#Database
# x = Data, y = Target
x = [[4],[8],[12],[16],[20],[24],[28],[32],[36],[40],[44],[48],[52],[56],[60],[64],[70],[74],[78],[82]]
y = [8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,140,148,156,164]

#Data Uji
#predict = np.array([[12.86]])
if __name__ == '__main__':
    while 1:
        print("Prediksi")
        predict = input("Input Prediksi: ")
        predict = np.array([[predict]])
        poly = PolynomialFeatures(degree=2)
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)
        x_ = poly.fit_transform(x)
        predict_ = poly.fit_transform(predict)
        regr = linear_model.LinearRegression()
        regr.fit(x_,y)
        #Menampilkan data prediksi
        #print("Prediksi")
        #print("Input = ", predict)
        #print("Output = ", regr.predict(predict_))
        print("Output = ", regr.predict(predict_).astype(int), "\n----------------------------")
