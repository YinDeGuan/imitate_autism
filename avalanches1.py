#



#********************************************************
#
# jieba example 1 
#
if __name__ == '__main__1' : 

    import jieba
    import os 


    seg_list = jieba.cut('小明1995生于北京清华大学',cut_all=False)
    print("Default Mode:"," ".join(seg_list))

    seg_list = jieba.cut('小明1995生于北京清华大学')
    print(' '.join(seg_list))

    seg_list = jieba.cut('小明1995生于北京清华大学',cut_all=True)
    print('\\'.join(seg_list))

    seg_list = jieba.cut_for_search('小明1995生于北京清华大学')
    print('/'.join(seg_list))







#**************************************************
#
# jieba example 2 
#
#

if __name__ == '__main__2' : 

    import os 
    import jieba 


    def savefile(path,content) :
        with open(path,'wb') as fp : # write and binary mode 
            fp.write(content)

    def readfile(path) :
        with open(path,'rb') as fp : # read and binary mode 
            re = fp.read()
            #can able to be read character 
        return re 

    corpus_path = 'D:/dieLuftDerFreiheitWeht/profile/creatist/ml/imitate_autism/corpus_un/'
    seg_path = 'D:/dieLuftDerFreiheitWeht/profile/creatist/ml/imitate_autism/corpus_en/' 
    dirlist = os.listdir(corpus_path) 

    for myd in dirlist : 
        sourcepath = corpus_path+myd+'/' 
        goalpath = seg_path+myd+'/'
        if not os.path.exists(goalpath) :
            os.makedirs(goalpath)

        filelist = os.listdir(sourcepath) # level 2 directory

        for f in filelist :
            sourcef = sourcepath+f
            goalf = goalpath+f
            cont = readfile(sourcef).strip()
            cont = cont.decode().replace('\r\n','').strip()
            #decode  to be string 
            cont_seg = jieba.cut(cont)
            savefile(goalf,' '.join(cont_seg).encode())
            #encode  to be bytes 

    print("Chinese segmentation done ")



#***********************************************
# data persistence
#
if __name__ == '__main__' : 
    
    from sklearn.datasets._base import Bunch
    import pickle
    import os 
    

    def readfile(path) :
        with open(path,'rb') as fp : # read and binary mode 
            re = fp.read()
            #can able to be read character 
        return re 



    wordbag_path = "D:/dieLuftDerFreiheitWeht/profile/creatist/ml/imitate_autism/train_set.dat"
    seg_path = 'D:/dieLuftDerFreiheitWeht/profile/creatist/ml/imitate_autism/corpus_en/' 

    catelist = os.listdir(seg_path) 

    bunch = Bunch(target_name = [] , label =[] , filenames=[] , contents=[]) 

    bunch.target_name.extend(catelist)
    #save category information 

    for label in catelist :
        classpath = seg_path + label +"/"
        filelist = os.listdir(classpath)
        for f in filelist : 
            fullname = classpath + f 
            bunch.label.append(label)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname).strip())
    
    with open(wordbag_path , "wb") as f :
        pickle.dump(bunch,f)
        #persistence
    
    print("bunch done ")
