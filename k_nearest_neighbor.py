
#**********************************************************
# purpose : debug knn method base on text classification 
# coder : liGil
# version : 0.1
#
# specification : 
#   distance 's implication to knn is G point 
#**************************************************************

import numpy as np 
import matplotlib.pyplot as plt 
import operator as ope
import naive_bayes as nb 

k = 3  # 3 point resolve 

#*************************************************************
# basic principle debug 
#

def DataSet() :
    group = np.array([[1.0,1.1],[1.0,1.0],[0.0,0.2],[0,0.1]])
    labels = ['A',  'A' , 'B' , 'B']
    return group , labels  

def SomePoint() : 
    data , labels = DataSet()
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    indx = 0 
    for point in data :
        if labels[indx] == 'A' :
            ax.scatter(point[0],point[1],c='blue',marker='o',
            linewidths=0,s=15)
            plt.annotate("("+str(point[0])+","+
            str(point[1])+")",xy=(point[0],point[1]))
        else :
            ax.scatter(point[0],point[1],c='red',marker='^',
            linewidths=0,s=15)
            plt.annotate("("+str(point[0])+","+
            str(point[1])+")",xy=(point[0],point[1]))
        indx += 1 
    plt.show()




#******************************************************************************
#
# distance base on cosin
# variety distance method bring difference result , 
# and here be one prefer to text classify .  
#
#
def cosDist(vector1 , vector2 ) :
    return np.dot(vector1, vector2) / np.linalg.norm(vector1) * np.linalg.norm(vector2) 



#*******************************************************************
#
# knn classfier 
#
def classify(testData , trainSet, listClasses , k ) :
    docLength = trainSet.shape[0] #trainSet , a Array object 
    distance = np.array(np.zeros(docLength)) 
    for index in range(docLength) :
        distance[index] = cosDist(testData , trainSet[index])
    sortedDistanceIndices = np.argsort(-distance) 
    classCount = {}
    for i in range(k) : # to extract k that has max cosin 
        label = listClasses[sortedDistanceIndices[i]]
        classCount[label] = classCount.get(label , 0 ) + 1 # class weight  

    sortedClassCount = sorted(classCount.items() , 
        key = ope.itemgetter(1) , reverse = True ) # sort base on dict's value 
    return sortedClassCount[0][0] # knn resolve's class 


if __name__ == '__main__no_debug' :
    
    #SomePoint()
    dataSet , listClasses = nb.loadDataSet() 
    inb = nb.naive_bayes()
    inb.train_set(dataSet , listClasses)
    print(classify( inb.tf[3] , inb.tf , listClasses , k ))
    
    #output : 1 
    # deserved surface opinion

