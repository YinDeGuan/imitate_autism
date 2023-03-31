#

import os 
import numpy as np




#**********************************************
# aim at like iris.cvs form file 
# 
def file2matrix(path , delimiter) :
    recordlist = []
    fp = open(path , 'r')
    content = fp.read()  #whole content readed 
    fp.close()
    rowlist = content.splitlines()[1:] 
    # splite lines to elements 

    recordlist = \
    [ list(map(eval , row.split(delimiter)[3:5])) for row in rowlist if row.strip()]
    # eval would transform str to number 
    # strip should remove ' ' or '\t'
    # every map make a list here 
    #  
    return np.mat(recordlist) 
    # non 1 line matrix 




#*************************************************
# debug part 
if __name__ == "__main__1" :
    m = file2matrix("D:/dieLuftDerFreiheitWeht/profile/creatist/ml/imitate_autism/iris.csv" , ',')
    print(np.shape(m))
    col = min( m[:,1])
    print(col)