# import bayes
# import data_synthesizer
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class dataPrep:

    def __init__(self,df):

        self.df = df
        self.dfz = self.dataAdjusting()
        #self.toSynthesizer()
        self.X = self.dfz.iloc[:, :14].values
        self.Y = self.dfz.iloc[:, 14:15].values
        self.X_Set,self.Y_Set = self.dataProcessing()
        self.val_size = 0.1
        self.test_size = 0.1
        self.X_train, self.X_val, self.X_test, self.Y_train ,self.Y_val, self.Y_test = self.split_train_test()

    #### This was prepared to calculate correlation between the features
    def correl(self):
        dfz = self.df.copy()
        del dfz['TenYearCHD']
        coorz = dfz.corr()
        plt.pcolor(coorz)
        plt.yticks(np.arange(0.5, len(coorz.index), 1), coorz.index)
        plt.xticks(np.arange(0.5, len(coorz.columns), 1), coorz.columns)
        plt.xticks(rotation=45)
        plt.show()
        print(dfz.corr())

    #### This was prepared to calculate correlation between the features based on binary variables (X >> 1 > X = 1)
    def correlNormalized(self):
        dfz = self.df.copy()
        del dfz['TenYearCHD']
        dfz[dfz > 1] = 0.999999 ## if 1 then we get nan, lowred to .999999
        coorz = dfz.corr()
        plt.pcolor(coorz)
        plt.yticks(np.arange(0.5, len(coorz.index), 1), coorz.index)
        plt.xticks(np.arange(0.5, len(coorz.columns), 1), coorz.columns)
        plt.xticks(rotation=45)
        plt.show()
        print(dfz.corr())

    def dataAdjusting(self):
        dfz = self.df
        del dfz['currentSmoker'] # this was the only column deleted
        return dfz

    def dataProcessing(self):
        sc = StandardScaler()
        ohe = OneHotEncoder()
        X_Set = sc.fit_transform(self.X)
        # Y_Set = ohe.fit_transform(self.Y).toarray()
        Y_Set = self.Y.squeeze()
        return X_Set, Y_Set

    def split_train_test(self):
        X_train,X_remain,Y_train,Y_remain = train_test_split(self.X_Set,self.Y_Set,test_size=(self.val_size + self.test_size))
        new_test_size = np.around(self.test_size / (self.val_size + self.test_size), 2)
        # To preserve (new_test_size + new_val_size) = 1.0
        new_val_size = 1.0 - new_test_size
        X_val, X_test ,Y_val, Y_test= train_test_split(X_remain,Y_remain, test_size=new_test_size)
        return X_train, X_val, X_test, Y_train ,Y_val, Y_test

    def toSynthesizer(self):
        dfzS = self.dfz
        dfzS.to_csv('modules/model.csv',index=',')

    def returnData(self):
        return [self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test]