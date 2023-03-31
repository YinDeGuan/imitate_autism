#

import p_tool
import numpy as np
import matplotlib.pyplot as plt 


#**********************************************
#
#
#can be used to n dimention data . 
# but private draw function merely to use by 2 dimention data 
#
class Kmeans(object) :
    def __init__(self , data_set , k) :
        self.data_set = data_set
        self.k = k
        self.m = np.shape(self.data_set)[0]  
        self.n = np.shape(self.data_set)[1]  #column number 

        self.clusterDist =  np.mat(np.zeros((self.m,2))) 
        self.clusterCenter = 0 

        self._randomCenters()


    def _randomCenters(self) :
        self.clusterCenter = np.mat(np.zeros( (self.k , self.n)))  

        for col in range(self.n) :
            mincol = min( self.data_set[:,col]) 
            maxcol = max( self.data_set[:,col])
            self.clusterCenter[:,col] =  np.mat(mincol + float(maxcol-mincol) * np.random.rand( self.k , 1))
            

    def _distEclud(self , i) :
        return [ 
            np.linalg.norm( self.clusterCenter[j,:] - self.data_set[i,:])
            for j in range(self.k)]


    def _color_cluster(self) : 
        idx2 = 0 
        indexmem = self.clusterDist[:,0:1]
        for idx1 in range(self.m) :
            if indexmem[idx1] == 0 :
                plt.scatter(self.data_set[idx2,0] ,
                self.data_set[idx2, 1] , c='blue' , marker='o')
            elif indexmem[idx1] == 1 :
                plt.scatter(self.data_set[idx2,0] ,
                self.data_set[idx2, 1] , c='green' , marker='o')
            elif indexmem[idx1] == 2 :
                plt.scatter(self.data_set[idx2,0] ,
                self.data_set[idx2, 1] , c='red' , marker='o')
            idx2 += 1


    def _proto_data_set(self) :
        plt.scatter( self.data_set[:,0].tolist() , self.data_set[:,1].tolist() ,
        c='red' )
        plt.show() 


    def _drawCenter(self) :
        plt.scatter( self.clusterCenter.T[0].tolist() , self.clusterCenter.T[1].tolist() ,
        s=20 , c='black' , marker = 'D')

    def _drawResult(self) :
        self._color_cluster()
        self._drawCenter()
        plt.show() 


    def cluster(self) : 
        flag = True
        
        while flag :
        
            flag = False 
        
            for i in range(self.m) : 
                distlist = self._distEclud(i)
                minDist = min(distlist)
                minIndex = distlist.index(minDist)

                if self.clusterDist[i,0] != minIndex :
                    flag = True 

                self.clusterDist[i,:] = minIndex , minDist 

            for cent in range(self.k) : 
                clustered_first_col = self.data_set[
                    np.nonzero(self.clusterDist[:,0] == cent)[0]]
                #self.clusterDist[:,0] like self.clusterDist[:,0].A by test 
                #so that be done
                #
                
                self.clusterCenter[cent,:]=np.mean(clustered_first_col,axis=0)
        
        return self.clusterCenter , self.clusterDist       



if __name__ == '__main__' :

    print(' test to kmeans ')

    data_set = p_tool.file2matrix("D:/dieLuftDerFreiheitWeht/profile/creatist/ml/imitate_autism/iris.csv" , ',')    
    km = Kmeans(data_set , k=3 )
    #km._proto_data_set()

    km.cluster() 
    km._drawResult()
    #how can arrange to index 
