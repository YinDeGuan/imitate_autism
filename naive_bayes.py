
#***************************************************************
#
#purpose : debug naive_bayes base on text classification
#coder : liGil
#version : 0.1
#
#*****************************************************************

import numpy as np
#from sklearn.model_selection import train_test_split 

class naive_bayes(object) :
    

    def __init__(self) :
        self.vocabulary = [] 

        self.idf = 0            # weight of idf of dict
        self.tf = 0             # weight matnirix of train sets
                                # weight value tf * idf can replace with frequency 

        self.tdm = 0            # P(x|y_i) 相对于一个划分的条件概率
        self.pCates = {}        # P(y_i) 是一个类别字典 , 划分的概率
        self.labels = []        # corresponding to class of per text ,
                                # a list from external importation

        self.docLength = 0      # doc amount of train sets
        self.vocabLen = 0       # vocabulary length of dict
        self.testset = 0        # test set


    
    #*********************************************************
    #
    #
    def train_set(self , trainset ,classVec) :
        self.cate_prob(classVec)    # cal P(y_i) 
        self.docLength = len(trainset) # the number of files 
        
        tempset = set() 
        [ tempset.add(word) for doc in trainset for word in doc ] #create dict
        self.vocabulary = list(tempset) 
                                #eliminate repetition words

        self.vocabLen =  len(self.vocabulary)
        self.calc_wordfreq(trainset)    #cal word frequence
        self.build_tdm()     #preparation for cal p(x|yi) 



    #******************************************************
    # to calcuate probablity about labels base on frequence
    #
    # 
    #   
    def cate_prob(self , classVec) :
        self.labels = classVec 
        labeltemps = set(self.labels)
        for labeltemp in labeltemps :
            self.pCates[labeltemp] = \
                float(self.labels.count(labeltemp)) \
                    /float(len(self.labels))
    


    #******************************************************
    #
    # tf and idf cal 
    # here's idf unused .   
    # merely for tf that base on frequence to classify 
    #
    def calc_wordfreq(self , trainset) :
        self.idf = np.zeros([1,self.vocabLen]) #we see later 
        self.tf = np.zeros([self.docLength , self.vocabLen])
        for indx in range(self.docLength) :
            for word in trainset[indx] :
                self.tf[indx , self.vocabulary.index(word)] += 1
            for signleword in set(trainset[indx]) :
                self.idf[0 , self.vocabulary.index(signleword)] +=1
                # that elements of idf list is more big means the elements have low 
                #  distinguishability . 
                #  



    #***********************************************************
    #
    # improvement frequence to tf*idf weight 
    #
    #
    def calc_tfidf(self , trainset) :
        self.idf = np.zeros([1 , self.vocabLen])
        self.tf = np.zeros([self.docLength , self.vocabLen])
        for indx in range(self.docLength) :
            for word in trainset[indx] : 
                self.tf[indx , self.vocabulary.index(word)] += 1
            self.tf[indx] = self.tf[indx]/float(len(trainset[indx])) 
            #eliminate difference with length of file 
            # 
            for signleword in set(trainset[indx]) :
                self.idf[0 , self.vocabulary.index(signleword)] +=1

        self.idf = np.log( float(self.doclength)/self.idf )
        self.tf = np.multiply( self.tf , self.idf )
        # with weight like tf * idf         



    #**************************************************
    #
    # preparation for cal p(x|yi)
    # G point 
    # 
    def build_tdm(self) :

        self.tdm = np.zeros( [len( self.pCates) , self.vocabLen])
        sumlist = np.zeros( [len(self.pCates) , 1] )

        for index in range(self.docLength) :
            self.tdm[ self.labels[index]] += self.tf[index]         
            sumlist[ self.labels[index]] = np.sum(self.tdm[self.labels[index]])
        self.tdm = self.tdm / sumlist


    #*******************************************************
    # map to vocab 
    def map2vocab(self , testdata) :
        self.testset = np.zeros( [1 , self.vocabLen])
        for word in testdata : 
            self.testset[ 0 , self.vocabulary.index(word)] += 1 


    #***********************************************************
    #
    def predict(self) :
        if np.shape(self.testset)[1] != self.vocabLen :
            print("testset illegal")
            exit(0)
        preValue = 0
        preClass = ""
        for tdmVect , keyClass in zip( self.tdm , self.pCates) :
            temp = np.sum( self.testset * tdmVect * self.pCates[keyClass])
            if temp > preValue :
                preValue = temp
                preClass = keyClass 
        return preClass 




def loadDataSet() :
    postingList = [
        ['my' , 'dog' , 'has' , 'flea' , 'problem' , 'help' ,'please'],
        ['maybe' , 'not' , 'take' , 'him' , 'to' , 'dog' , 'park' , 'stupid'] ,
        ['my' , 'daimation' , 'is' , 'so' , 'cute' , 'I' , 'love' , 'him' , 'my'] ,\
        ['stop' , 'posting' , 'stupid' , 'worthless' , 'garbage'] , 
        ['mr' , 'licks' , 'ate' , 'my' , 'steak' , 'how' , 'to' , 'stop' , 'him'] , 
        ['quit' , 'buying' , 'worthless', 'dog' ,'food' , 'stupid']]
        # a line is a file 
        
    classes = [0 , 1 , 0 , 1 , 0 , 1] 
    # 1 is abusive , 0 not 
    # like the kind of comment 
    return postingList , classes 


if __name__ == '__main__no_debug' :

    dataset , listClasses = loadDataSet()
    nb = naive_bayes()
    nb.train_set(dataset , listClasses) 

    nb.map2vocab(dataset[0])
    print(nb.predict())

    
    
