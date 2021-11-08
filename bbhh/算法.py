import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
iris=pd.read_csv('iris_data.csv')
#sl:花瓣长，sw:花瓣宽
print(iris)
fig=iris[iris.leixing=='Iris-setosa'].plot(kind='scatter',x='sl',y='sw',color='orange',label='0')
iris[iris.leixing=='Iris-versicolor'].plot(kind='scatter',x='sl',y='sw',color='blue',label='1',ax=fig)
iris[iris.leixing=='Iris-virginica'].plot(kind='scatter',x='sl',y='sw',color='green',label='2',ax=fig)
fig.get_xlabel()
fig.get_ylabel()
fig.set_title('leng_vs')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
#pl:花萼长，pw:花萼宽
fig1=iris[iris.leixing=='Iris-setosa'].plot(kind='scatter',x='pl',y='pw',color='orange',label='0')
iris[iris.leixing=='Iris-versicolor'].plot(kind='scatter',x='pl',y='pw',color='blue',label='1',ax=fig1)
iris[iris.leixing=='Iris-virginica'].plot(kind='scatter',x='pl',y='pw',color='green',label='2',ax=fig1)
fig1.get_xlabel()
fig1.get_ylabel()
fig1.set_title('width_vs')
fig1=plt.gcf()
fig1.set_size_inches(8,4)
plt.show()
iris.hist(edgecolor='black',linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
iris.drop('label',axis=1,inplace=True)#axis=0删除行
iris.shape
plt.figure(figsize=(7,4))
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')#相关性图
plt.show()#相关性图

train,test=train_test_split(iris,test_size=0.3)
print('训练集大小'+str(train.shape))
print('测试集大小'+str(test.shape))
#以下是训练、测试部分
train_X=train[['sl','sw','pl','pw']]
#leixing：花的类型
train_y=train.leixing
test_X= test [['sl','sw','pl','pw']]
test_y=test.leixing
#print(train_X.head(2))
test_X.head(2)
train_y.head(2)
model=svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)



#plt.scatter()会根据y_train的结果自动的在cmap中选择颜色，c参数代表颜色

print("svm精准率是："+str(metrics.precision_score(prediction,test_y,average=None)))
print("svm召回率是："+str(metrics.recall_score(prediction,test_y,average=None)))
print("svm准确度是："+str(metrics.accuracy_score(prediction,test_y)))
print("svm调和平均是："+str(metrics.f1_score(prediction,test_y,average=None)))
#逻辑回归
print("/////"
      "/////")
model1=LogisticRegression()
model1.fit(train_X,train_y)
prediction1=model1.predict(test_X)
print("逻辑回归精准率是："+str(metrics.precision_score(prediction1,test_y,average=None)))
print("逻辑回归召回率是："+str(metrics.recall_score(prediction1,test_y,average=None)))
print("逻辑回归准确度是："+str(metrics.accuracy_score(prediction1,test_y)))
print("逻辑回归调和平均是："+str(metrics.f1_score(prediction1,test_y,average=None)))
#决策树
print("/////"
      "/////")
model2=DecisionTreeClassifier()
model2.fit(train_X,train_y)
prediction2=model2.predict(test_X)
print("决策树精准率是："+str(metrics.precision_score(prediction2,test_y,average=None)))
print("决策树召回率是："+str(metrics.recall_score(prediction2,test_y,average=None)))
print("决策树准确度是："+str(metrics.accuracy_score(prediction2,test_y)))
print("决策树调和平均是："+str(metrics.f1_score(prediction2,test_y,average=None)))
