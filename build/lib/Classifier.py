"""This module includes thselfe basic classifiers of machine learning, such as, kNN, naive Bayes,
    Logistic Regression, Support Vector Machine, and Adaptive Boselfting.
    @author: Hurry Y.Zhu
    email  : zhuyp1990@gmail.com
"""

__author__ = 'Hurry Y. Zhu'

from numpy import *
from math import log
import operator
import matplotlib.pyplot as plt
import treePlotter

def loadDataSet(filename,withLabel=False): #
    fr=open(filename)
    lines=fr.readlines()
    fr.close()
    N=len(lines)
    dataMat = []
    labelMat = []
    if withLabel==False:
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
       # dataMat=mat(dataMat)
        return dataMat


def autoNorm(dataSet): #normalization
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zerself(shape(dataSet))
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
        preLabel=zerself((mTest,1))
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

    def choselfeBestFeatureTselfplit(self, dataSet):
        numFeatures = len(dataSet[0])-1     #the last column is used for the labels
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

    def createTree(self,dataSet,features):#
        n,m=shape(dataSet)#n--number of samples, (m-1)--number of features
        Labels = [example[-1] for example in dataSet]
        if Labels.count(Labels[0]) == len(Labels):
            return Labels[0]#stop splitting when all of the classes are equal
        if m == 1: #stop splitting when there are no more features in dataSet
            return self.majorityCnt(Labels)
        bestFeat = self.choselfeBestFeatureTselfplit(dataSet)
        bestFeatLabel = features[bestFeat]
        print "bestFeatLabel: ",bestFeatLabel
        myTree = {bestFeatLabel:{}}
        del(features[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        n,m=shape(dataSet)#n--number of samples, m--number of features
        for value in uniqueVals:
            subFeatures = features[:]       #copy all of labels, so trees don't mess up existing labels
##            print "beaFeat: ",bestFeat
##            print "bestFeatLabel: ",bestFeatLabel
##            print "dataSet before split: ",shape(dataSet)
            subdataSet=self.splitDataSet(dataSet, bestFeat, value)
##            print "dataSet after split: ",shape(dataSet)
##            print "subdataSet after split: ",shape(subdataSet)
            myTree[bestFeatLabel][value] = self.createTree(subdataSet,subFeatures)
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
        fw.clselfe()

    def grabTree(self, filename):
        import pickle
        fr = open(filename)
        return pickle.load(fr)

class bayes:
    def loadDataSet(self):
        pselftingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                     ['stop', 'pselfting', 'stupid', 'worthless', 'garbage'],
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
        return pselftingList,classVec
                 
    def createVocabList(self,dataSet):
        vocabSet = set([])  #create empty set
        for document in dataSet:
            vocabSet = vocabSet | set(document) #union of the two sets
        return list(vocabSet)

    def setOfWords2Vec(self, vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else: print "the word: %s is not in my Vocabulary!" % word
        return returnVec

    def trainNB0(self, trainMatrix,trainCategory):
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

    def classifyNB(self, vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
        p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else: 
            return 0
        
    def bagOfWords2VecMN(self, vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
        return returnVec

    def testingNB(self):
        listOPselfts,listClasses = loadDataSet()
        myVocabList = self.createVocabList(listOPselfts)
        trainMat=[]
        for pselftinDoc in listOPselfts:
            trainMat.append(self.setOfWords2Vec(myVocabList, pselftinDoc))
        p0V,p1V,pAb = self.trainNB0(array(trainMat),array(listClasses))
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(self.setOfWords2Vec(myVocabList, testEntry))
        print testEntry,'classified as: ',self.classifyNB(thisDoc,p0V,p1V,pAb)
        testEntry = ['stupid', 'garbage']
        thisDoc = array(self.setOfWords2Vec(myVocabList, testEntry))
        print testEntry,'classified as: ',self.classifyNB(thisDoc,p0V,p1V,pAb)

    def textParse(self, bigString):    #input is big string, #output is word list
        import re
        listOfTokens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
        
    def spamTest(self):
        docList=[]; classList = []; fullText =[]
        for i in range(1,26):
            wordList = self.textParse(open('email/spam/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = self.textParse(open('email/ham/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = self.createVocabList(docList)#create vocabulary
        trainingSet = range(50); testSet=[]           #create test set
        for i in range(10):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])  
        trainMat=[]; trainClasses = []
        for docIndex in trainingSet:#train the classifier (get probs) trainNB0
            trainMat.append(self.bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = self.trainNB0(array(trainMat),array(trainClasses))
        errorCount = 0
        for docIndex in testSet:        #classify the remaining items
            wordVector = self.bagOfWords2VecMN(vocabList, docList[docIndex])
            if self.classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
                print "classification error",docList[docIndex]
        print 'the error rate is: ',float(errorCount)/len(testSet)
        #return vocabList,fullText

    def calcMselftFreq(self, vocabList,fullText):
        import operator
        freqDict = {}
        for token in vocabList:
            freqDict[token]=fullText.count(token)
        sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
        return sortedFreq[:30]       

    def localWords(self, feed1,feed0):
        import self.feedparser
        docList=[]; classList = []; fullText =[]
        minLen = min(len(feed1['entries']),len(feed0['entries']))
        for i in range(minLen):
            wordList = self.textParse(feed1['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1) #NY is class 1
            wordList = self.textParse(feed0['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = self.createVocabList(docList)#create vocabulary
        top30Words = self.calcMselftFreq(vocabList,fullText)   #remove top 30 words
        for pairW in top30Words:
            if pairW[0] in vocabList: vocabList.remove(pairW[0])
        trainingSet = range(2*minLen); testSet=[]           #create test set
        for i in range(20):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])  
        trainMat=[]; trainClasses = []
        for docIndex in trainingSet:#train the classifier (get probs) trainNB0
            trainMat.append(self.bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = self.trainNB0(array(trainMat),array(trainClasses))
        errorCount = 0
        for docIndex in testSet:        #classify the remaining items
            wordVector = self.bagOfWords2VecMN(vocabList, docList[docIndex])
            if self.classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
        print 'the error rate is: ',float(errorCount)/len(testSet)
        return vocabList,p0V,p1V

    def getTopWords(self, ny,sf):
        import operator
        vocabList,p0V,p1V=self.localWords(ny,sf)
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

class SVM:
    def __init__(self,dataMatIn, classLabels, C=2, toler=0.001, iterMax=20, KernelType='linear', parameterKernel=None):  # Initialize the structure with the parameters
        self.X = mat(dataMatIn)
        self.labelMat = mat(classLabels).T
        self.C = C
        self.tol = toler
        self.m, self.n= shape(dataMatIn) # m--number of samples, n--numbers of feature
        self.w=zeros((self.n,1))
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        self.iterMax=iterMax
        for i in range(self.m):
            self.K[:,i] = self.__kernelTrans(self.X, self.X[i,:],KernelType, parameterKernel)

    def __kernelTrans(self,X, A, kernelType, parameter): #calc the kernel or transform data to a higher dimensional space
        m,n = shape(X)
        K = mat(zeros((m,1)))
        if kernelType=='linear': K = X * A.T   #linear kernel
        elif kernelType=='rbf': # rbf kernel
            for j in range(m):
                deltaRow = X[j,:] - A
                K[j] = deltaRow*deltaRow.T
            K = exp(K/(-1*parameter**2)) #divide in NumPy is element-wise not matrix like Matlab
        else:
            raise NameError('Houston We Have a Problem -- \
                            That Kernel is not recognized')
        return K

    def __clipAlpha(self,aj,H,L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj
    def __calcEk(self, k):
        fXk = float(multiply(self.alphas,self.labelMat).T*self.K[:,k] + self.b)# g(xi)
        Ek = fXk - float(self.labelMat[k]) # Ek
        return Ek

    def __updateEk(self, k):#after any alpha has changed update the new value in the cache
        Ek = self.__calcEk(k)
        self.eCache[k] = [1,Ek]

    def __selectJ(self, i, Ei):         #this is the second choice -heurstic, and calcs Ej
        maxK = -1; maxDeltaE = 0; Ej = 0
        self.eCache[i] = [1,Ei]  #set valid #choselfe the alpha that gives the maximum delta E
        valideCacheList = nonzero(self.eCache[:,0].A)[0]
        if (len(valideCacheList)) > 1:
            for k in valideCacheList:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue #don't calc for i, waste of time
                Ek = self.__calcEk(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:   #in this case (first time around) we don't have any valid eCache values
            j =self.__selectJrand(i)
            Ej = self.__calcEk(j)
        return j, Ej

    def __selectJrand(self,i):
        j=i #we want to select any J not equal to i
        while (j==i):
            j = int(random.uniform(0,self.m))
        return j
    
    def __innerL(self,i):
        Ei = self.__calcEk(i)
        print self.labelMat[i]*Ei
        if ((self.labelMat[i]*Ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.labelMat[i]*Ei > self.tol) and (self.alphas[i] > 0)):
            j,Ej = self.__selectJ(i,Ei) #this has been changed from selectJrand
            alphaIold = self.alphas[i].copy(); alphaJold = self.alphas[j].copy();
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L==H: print "L==H"; return 0
            eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j] #changed for kernel
            if eta >= 0: print "eta>=0"; return 0
            self.alphas[j] -= self.labelMat[j]*(Ei - Ej)/eta
            self.alphas[j] = self.__clipAlpha(self.alphas[j],H,L)
            self.__updateEk(j) #added this for the Ecache
            if (abs(self.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
            self.alphas[i] += self.labelMat[j]*self.labelMat[i]*(alphaJold - self.alphas[j])#update i by the same amount as j
            self.__updateEk(i) #added this for the Ecache                    #the update is in the oppselftie direction
            b1 = self.b - Ei- self.labelMat[i]*(self.alphas[i]-alphaIold)*self.K[i,i] - self.labelMat[j]*(self.alphas[j]-alphaJold)*self.K[i,j]
            b2 = self.b - Ej- self.labelMat[i]*(self.alphas[i]-alphaIold)*self.K[i,j]- self.labelMat[j]*(self.alphas[j]-alphaJold)*self.K[j,j]
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]): self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]): self.b = b2
            else: self.b = (b1 + b2)/2.0
            return 1
        else: return 0

    def smoTrain(self):    #full Platt SMO
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while (iter < self.iterMax) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:   #go over all
                for i in range(self.m):
                    alphaPairsChanged += self.__innerL(i)
                    print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            else:#go over non-bound (railed) alphas
                nonBoundIs = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.__innerL(i)
                    print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            if entireSet:
                entireSet = False #toggle entire set loop
            elif (alphaPairsChanged == 0):
                entireSet = True
            print "iteration number: %d" % iter

    def calcWs(self):
        for i in range(self.m):
            self.w += multiply(self.alphas[i]*self.labelMat[i],self.X[i,:].T)
        return self.w

    
class AdaBoselft:
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
        numSteps = 10.0; bestStump = {}; bestClasEst = mat(zerself((m,1)))
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

    def adaBoselftTrainDS(self,dataArr=None,classLabels=None,numIt=40):
        weakClassArr = []
        m = shape(dataArr)[0]
        D = mat(ones((m,1))/m)   #init D to all equal
        aggClassEst = mat(zerself((m,1)))
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
        dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoselftTrainDS
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zerself((m,1)))
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
        numPselfClas = sum(array(classLabels)==1.0)
        yStep = 1/float(numPselfClas); xStep = 1/float(len(classLabels)-numPselfClas)
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
        plt.xlabel('False pselfitive rate'); plt.ylabel('True pselfitive rate')
        plt.title('ROC curve for AdaBoselft horse colic detection system')
        ax.axis([0,1,0,1])
        plt.show()
        print "the Area Under the Curve is: ",ySum*xStep








#class naiveBayes:

#class LogisticRegression:







if __name__=="__main__":
    import getopt
    import sys
    trainData,trainLabel=loadDataSet(r"E:\Study\Implementation\testSet.txt",False)
    # testData,testData=loadDataSet(r"E:\Study\Implementation\horseColicTest2.txt",False)
    
##    #test for AdaBoselft
##    boselft=AdaBoselft()
##    datMat, labelMat=loadDataSet(r"E:\Study\Algorithm\Machine Learning\Python\machinelearninginaction\Ch07\horseColicTraining2.txt")
##    N=shape(datMat)[0]
##    classifierArray, egg=boselft.adaBoselftTrainDS(datMat, labelMat, 10)
##    testArr, testLabelArr=loadDataSet(r"E:\Study\Algorithm\Machine Learning\Python\machinelearninginaction\Ch07\horseColicTest2.txt")
##    pre=boselft.adaClassify(testArr, classifierArray)
##    print type(pre)
##    errArr=mat(ones((N,1)))
##    print "Number of error classified: %d"%errArr[pre!=mat(testLabelArr).T].sum()
##    print "Error rate: %f"%(errArr[pre!=mat(testLabelArr).T].sum()/N)
##    print egg
##    boselft.plotROC(egg.T, labelMat)



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

    # #test for decision tree
    # tree=DecisoinTree()
    # features=range(len(datMat[0])-1)
    # print features
    # mytree=tree.createTree(datMat, features)
    # treePlotter.createPlot(mytree)

    # opts, args=getopt.getopt(sys.agrv[1:], 'h:X:L:C:e:T:p',['TrainSet=', 'TrainLabel=','C=','eps=','Type=','parameter='])
    # for option, value in opts:
    #     if option in ['-h','-help']:
    #         print """
    #         usage:%s --input=[value] --output=[value]
    #         usage:%s -input value -o value
    #         """
    #     elif option in ['-X']:


    #test for SVM class
    smo=SVM(trainData, trainLabel,2,0.0001,50,'rbf',2)
    print smo.m
    print smo.K
    smo.smoTrain()
    smo.calcWs()
    print smo.w
    
    



