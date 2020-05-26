#导入必要的包
import numpy as np
import operator
from scipy.spatial import distance
#仿sklearn接口定义KNN类
class SdKNN():
    def __init__(self):
        pass

    def fit(self,x_train,y_train,k):
        """
        fit函数 参数
        x_train: 训练集特征
        y_train: 训练集标签
        k:  k个相邻的元素
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k=k
    def predict(self,x_test):
        """
        predict函数
        x_test: 测试集特征
        以list形式返回 预测值
        """
        predictions=[]
        for row in x_test:
            plabel=self.closest_k(row)
            predictions.append(plabel)
        return predictions

    def closest_k(self,row):
        """
        k个距离最近的元素
        row: 每个测试集的元素
        返回 最相近的一个分类的标签
        """
        # 将每个测试集中的元素与训练集的元素计算欧式距离存在distances列表中
        distances=[]
        for i in range(len(x_train)):
            dist=self.euc(row,self.x_train[i])
            distances.append(dist)
        #转成np.array类型
        distances=np.array(distances)
        #argsort函数返回的是距离值从小到大的索引值
        sortedDistIndicies=distances.argsort()

        #在距离最小的k个值中寻找投票数最多的分类
        #classCount存，对应的类别的统计个数
        classCount={}
        for i in range(self.k):
            voteIlabel=y_train[sortedDistIndicies[i]]
            #此处get,原字典有此voteIlabel则返回其对应的值否则返回0
            classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        #根据值（对应的类别的统计个数）进行排序，使得统计个数最多的排在前面
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        #返回测试点的类别
        return sortedClassCount[0][0]

    def euc(self,a,b):
        """
        调用scipy包中欧式距离函数
        """
        return distance.euclidean(a,b)



#测试
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target

from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)

my_classifier=SdKNN()
my_classifier.fit(x_train,y_train,k=3)
predictions=my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))



