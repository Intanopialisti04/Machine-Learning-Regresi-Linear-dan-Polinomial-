import numpy as np
from sklearn.linear_model import LinearRegression
#import warnings
from warnings import simplefilter
#Database
# x = Data, y = Target
x = [[4],[8],[12],[16],[20],[24],[28],[32],[36],[40],[44],[48],[52],[56],[60],[64],[70],[74],[78],[82]]
y = [8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,140,148,156,164]

regr = LinearRegression().fit(x,y)
regr.score(x,y)

#Data Uji
#predict = np.array([[3]])
if __name__ == '__main__':
    while 1:
        print("Prediksi")
        predict = input("Input Prediksi: ")
        predict = np.array([[predict]])
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)
        #Menampilkan data prediksi
        print("Output = ", regr.predict(predict).astype(int), "\n----------------------------")


