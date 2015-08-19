"""This module includes those basic classifiers of machine learning, such as, kNN, naive Bayes,
    Logistic Regression, Support Vector Machine, and Adaptive Boosting.
    @author: Hurry Y.Zhu
    email  : zhuyp1990@gmail.com
"""

__author__ = 'Hurry Y. Zhu'

from numpy import *
from math import log
import operator
import matplotlib.pyplot as plt

def loadDataSet(filename,withLabel=True): #
    fr=open(filename)
    lines=fr.readlines()
    fr.close()
    N=len(lines)
    dataMat = []
    labelMat = []
    if withLabel==True:
        for line in lines:
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(len(curLine)-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
##        dataMat=mat(dataMat)
##        labelMat=mat(labelMat).reshape(N,1)
        return dataMat,labelMat
    else:
        for line in lines:
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(len(curLine)):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
        dataMat=mat(dataMat)
        return dataMat


def autoNorm(dataSet): #normalization
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

class kNN:
    def __init__(self,K=10):
        self.K=K

    def __classify0(self, inX, dataSet, labels, k): #other methods for distance calculation
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSetSize,1)) - dataSet
        diffMat=array(diffMat)
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort()
        classCount={}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def kNN(self,trainDataSet, trainLabelSet,testDataSet):
        trainDataSet=mat(trainDataSet)
        N=len(trainLabelSet)
        trainLabelSet=mat(trainLabelSet).reshape(N,1)
        testDataSet=mat(testDataSet)
        normTrainMat, rangesTrain, minTrainVals = autoNorm(trainDataSet)
        normTestMat, rangesTest, minTestVals = autoNorm(testDataSet)
        mTrain = normTrainMat.shape[0];mTest = normTestMat.shape[0]
        preLabel=zeros((mTest,1))
        for i in range(mTest):
            preLabel[i,0]=self.__classify0(normTestMat[i,:], normTrainMat, trainLabelSet, self.K)
        return preLabel

class DecisoinTree:
    def __calcShannonEnt(self,dataSet):
        numEntries = len(dataSet)  # entry sample number
        labelCounts = {}
        for featVec in dataSet: #the the number of unique elements and their occurance
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob,2) #log base 2
        return shannonEnt

    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            reducedFeatVec=[]
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0])     #the last column is used for the labels
        baseEntropy = self.__calcShannonEnt(dataSet)
        bestInfoGain = 0.0; bestFeature = -1
        for i in range(numFeatures):        #iterate over all the features
            featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
            uniqueVals = set(featList)       #get a set of unique values
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.__calcShannonEnt(subDataSet)
            
            infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
##            if infoGain==0.0:
##                print "uniqueVals: ",uniqueVals
            if (infoGain > bestInfoGain):       #compare this to the best gain so far
                bestInfoGain = infoGain         #if better than current best, set to best
                bestFeature = i
        return bestFeature                      #returns an integer

    def majorityCnt(self, classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self,dataSet,features, Labels):#
        n,m=shape(dataSet)#n--number of samples, m--number of features
        classList = [example[0] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]#stop splitting when all of the classes are equal
        if m == 1: #stop splitting when there are no more features in dataSet
            return self.majorityCnt(Labels)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = features[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(features[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        n,m=shape(dataSet)#n--number of samples, m--number of features
        for value in uniqueVals:
            subFeatures = features[:]       #copy all of labels, so trees don't mess up existing labels
            print "beaFeat: ",bestFeat
            print "bestFeatLabel: ",bestFeatLabel
            print "dataSet before split: ",shape(dataSet)
            subdataSet=self.splitDataSet(dataSet, bestFeat, value)
            print "dataSet after split: ",shape(dataSet)
            print "subdataSet after split: ",shape(subdataSet)
            myTree[bestFeatLabel][value] = self.createTree(subdataSet,subFeatures,Labels)
        return myTree

    def classify(self, inputTree,featLabels,testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):
            classLabel = self.classify( valueOfFeat, featLabels, testVec)
        else: classLabel = valueOfFeat
        return classLabel

    def storeTree(self, inputTree,filename):
        import pickle
        fw = open(filename,'w')
        pickle.dump(inputTree,fw)
        fw.close()

    def grabTree(self, filename):
        import pickle
        fr = open(filename)
        return pickle.load(fr)

class bayes:
    def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
    def createVocabList(dataSet):
        vocabSet = set([])  #create empty set
        for document in dataSet:
            vocabSet = vocabSet | set(document) #union of the two sets
        return list(vocabSet)

    def setOfWords2Vec(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else: print "the word: %s is not in my Vocabulary!" % word
        return returnVec

    def trainNB0(trainMatrix,trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = sum(trainCategory)/float(numTrainDocs)
        p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
        p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        p1Vect = log(p1Num/p1Denom)          #change to log()
        p0Vect = log(p0Num/p0Denom)          #change to log()
        return p0Vect,p1Vect,pAbusive

    def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
        p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else: 
            return 0
        
    def bagOfWords2VecMN(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
        return returnVec

    def testingNB():
        listOPosts,listClasses = loadDataSet()
        myVocabList = createVocabList(listOPosts)
        trainMat=[]
        for postinDoc in listOPosts:
            trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
        p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
        print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
        testEntry = ['stupid', 'garbage']
        thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
        print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

    def textParse(bigString):    #input is big string, #output is word list
        import re
        listOfTokens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
        
    def spamTest():
        docList=[]; classList = []; fullText =[]
        for i in range(1,26):
            wordList = textParse(open('email/spam/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = textParse(open('email/ham/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = createVocabList(docList)#create vocabulary
        trainingSet = range(50); testSet=[]           #create test set
        for i in range(10):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])  
        trainMat=[]; trainClasses = []
        for docIndex in trainingSet:#train the classifier (get probs) trainNB0
            trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
        errorCount = 0
        for docIndex in testSet:        #classify the remaining items
            wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
            if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
                print "classification error",docList[docIndex]
        print 'the error rate is: ',float(errorCount)/len(testSet)
        #return vocabList,fullText

    def calcMostFreq(vocabList,fullText):
        import operator
        freqDict = {}
        for token in vocabList:
            freqDict[token]=fullText.count(token)
        sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
        return sortedFreq[:30]       

    def localWords(feed1,feed0):
        import feedparser
        docList=[]; classList = []; fullText =[]
        minLen = min(len(feed1['entries']),len(feed0['entries']))
        for i in range(minLen):
            wordList = textParse(feed1['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1) #NY is class 1
            wordList = textParse(feed0['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = createVocabList(docList)#create vocabulary
        top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
        for pairW in top30Words:
            if pairW[0] in vocabList: vocabList.remove(pairW[0])
        trainingSet = range(2*minLen); testSet=[]           #create test set
        for i in range(20):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])  
        trainMat=[]; trainClasses = []
        for docIndex in trainingSet:#train the classifier (get probs) trainNB0
            trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
        errorCount = 0
        for docIndex in testSet:        #classify the remaining items
            wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
            if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
        print 'the error rate is: ',float(errorCount)/len(testSet)
        return vocabList,p0V,p1V

    def getTopWords(ny,sf):
        import operator
        vocabList,p0V,p1V=localWords(ny,sf)
        topNY=[]; topSF=[]
        for i in range(len(p0V)):
            if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
            if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
        sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
        print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
        for item in sortedSF:
            print item[0]
        sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
        print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
        for item in sortedNY:
            print item[0]

class AdaBoost:
    def stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):#just classify the data
        retArray = ones((shape(dataMatrix)[0],1))
        if threshIneq == 'lt':
            retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:,dimen] > threshVal] = -1.0
        return retArray

    def buildStump(self,dataArr,classLabels,D):
        dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
        m,n = shape(dataMatrix)
        numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
        minError = inf #init error sum, to +infinity
        for i in range(n):#loop over all dimensions
            rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
            stepSize = (rangeMax-rangeMin)/numSteps
            for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
                for inequal in ['lt', 'gt']: #go over less than and greater than
                    threshVal = (rangeMin + float(j) * stepSize)
                    predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                    errArr = mat(ones((m,1)))
                    errArr[predictedVals == labelMat] = 0
                    weightedError = D.T*errArr  #calc total error multiplied by D
                    #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump,minError,bestClasEst

    def adaBoostTrainDS(self,dataArr=None,classLabels=None,numIt=40):
        weakClassArr = []
        m = shape(dataArr)[0]
        D = mat(ones((m,1))/m)   #init D to all equal
        aggClassEst = mat(zeros((m,1)))
        for i in range(numIt):
            bestStump,error,classEst = self.buildStump(dataArr,classLabels,D)#build Stump
            print "D:",D.T
            alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)                  #store Stump Params in Array
            print "classEst: ",classEst.T
            expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
            D = multiply(D,exp(expon))                              #Calc New D for next iteration
            D = D/D.sum()
            #calculate training error of all classifiers, if this is 0 quit for loop early (use break)
            aggClassEst += alpha*classEst
            print "aggClassEst: ",aggClassEst.T
            aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
            errorRate = aggErrors.sum()/m
            print "total error: ",errorRate
            if errorRate == 0.0: break
        return weakClassArr, aggClassEst

    def adaClassify(self,datToClass,classifierArr):
        dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m,1)))
        for i in range(len(classifierArr)):
            classEst = self.stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                     classifierArr[i]['thresh'],\
                                     classifierArr[i]['ineq'])#call stump classify
            aggClassEst += classifierArr[i]['alpha']*classEst
            print aggClassEst
        return sign(aggClassEst)

    def plotROC(self, predStrengths, classLabels):
        cur = (1.0,1.0) #cursor
        ySum = 0.0 #variable to calculate AUC
        numPosClas = sum(array(classLabels)==1.0)
        yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
        sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        #loop through all the values, drawing a line segment at each point
        for index in sortedIndicies.tolist()[0]:
            if classLabels[index] == 1.0:
                delX = 0; delY = yStep;
            else:
                delX = xStep; delY = 0;
                ySum += cur[1]
            #draw line from cur to (cur[0]-delX,cur[1]-delY)
            ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
            cur = (cur[0]-delX,cur[1]-delY)
        ax.plot([0,1],[0,1],'b--')
        plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
        plt.title('ROC curve for AdaBoost horse colic detection system')
        ax.axis([0,1,0,1])
        plt.show()
        print "the Area Under the Curve is: ",ySum*xStep


#class SVM:





#class naiveBayes:

#class LogisticRegression:







if __name__=="__main__":
    datMat, labelMat=loadDataSet(r"E:\Study\Algorithm\Machine Learning\Python\machinelearninginaction\Ch07\horseColicTraining2.txt")
    testArr, testLabelArr=loadDataSet(r"E:\Study\Algorithm\Machine Learning\Python\machinelearninginaction\Ch07\horseColicTest2.txt")


##    #test for AdaBoost
##    boost=AdaBoost()
##    datMat, labelMat=loadDataSet(r"E:\Study\Algorithm\Machine Learning\Python\machinelearninginaction\Ch07\horseColicTraining2.txt")
##    N=shape(datMat)[0]
##    classifierArray, egg=boost.adaBoostTrainDS(datMat, labelMat, 10)
##    testArr, testLabelArr=loadDataSet(r"E:\Study\Algorithm\Machine Learning\Python\machinelearninginaction\Ch07\horseColicTest2.txt")
##    pre=boost.adaClassify(testArr, classifierArray)
##    print type(pre)
##    errArr=mat(ones((N,1)))
##    print "Number of error classified: %d"%errArr[pre!=mat(testLabelArr).T].sum()
##    print "Error rate: %f"%(errArr[pre!=mat(testLabelArr).T].sum()/N)
##    print egg
##    boost.plotROC(egg.T, labelMat)



##   # test for kNN
##    knn=kNN(11)
##    result=knn.kNN(datMat,labelMat,testArr)
##    errCount=0
##    N=shape(result)[0]
##    print result
##    for i in range(shape(result)[0]):
##        print result[i,0],'\t',testLabelArr[i]
##    if result[i,0]!=testLabelArr[i]:
##        errCount +=1
##    print "The rate of correct prediction: ",1-float(errCount)/N

    #test for decision tree
    tree=DecisoinTree()
    features=range(len(datMat[0]))
    print features
    mytree=tree.createTree(datMat, features,labelMat)
    print "mytree: \n",mytree


