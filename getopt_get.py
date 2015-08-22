# -*- coding: cp936 -*-
import getopt
import sys 
from Classifier import *

if __name__=="__main__":
    config={
    "trainDataPath":"",
    "C":1,
    "toler":0.0001,
    "iterMax":20,
    "kernelType":'linear',
    "gama":1,
    }
    try:
        opts,args = getopt.getopt(sys.argv[1:],'hP:C:t:i:k:g:',['help','trainDataPath=','C=','toler=','iterMax=','kernelType=','gama='])
    except getopt.GetoptError:
        print "Please input <--help> for help."
        sys.exit()

    for option, value in opts:
        if option in ['-h','--help']:
            print """
            usage:%s --trainDataPath=[value] --C=[value]  --toler=[value]  --kernelType=[value] --gama=[value]
            usage:%s -P value -C value -t value -k value -g value
            """
        elif option in ['-P','--trainDataPath']:
            config['trainDataPath']=value
            
        elif option in ['-C','--C']:
            config['C']=float(value)

        elif option in ['-i','--iterMax']:
            config['iterMax']=int(value)
            
        elif option in ['-t','--toler']:
            config['toler']=float(value)
        elif option in ['-k','--kernelType']:
            config['kernelType']=value
        elif option in ['-g','--gama']:
            config['gama']=float(value)
    print config

    trainData,trainLabel=loadDataSet(config['trainDataPath'],False)
    testData,testLabel=loadDataSet(r'E:\Study\Implementation\testSetRBF2.txt',False)
    svm=SVM(trainData,trainLabel,config['C'],config['toler'],config['iterMax'],config['kernelType'],config['gama'])
    print "svm.C=",svm.C
    print "svm.iterMax=",svm.iterMax
    svm.smoTrain()
    svm.calcWs()
    print svm.w
    
##    print "alphas: ",svm.alphas
    svm.predict(testData,testLabel)
    print shape(nonzero(svm.alphas)[0])[1]















