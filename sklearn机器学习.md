# Sklearn机器学习:eagle:



## 1. 决策树

> 选用了红酒数据集进行说明

### 1.1 决策树分类模型使用

以下代码中的，Xtrain，Ytrain，Xtest，Ytest 都是从数据集上分出的数据，而并不是真实的数据

```python
# 模型实例化
clf = tree.DecisionTreeClassifier(criterion='entropy')
# 模型拟合使用
clf = clf.fit(Xtrain,Ytrain)
# 评分。这里的分数是对模型的精确度的评分，因为其中相当于是把Xtest进行拟合之后得出Ypredict然后再与Ytest进行比较
score = clf.score(Xtest,Ytest)
```

### 1.2 常用参数说明

1. 常用的参数

|参数|说明|使用|
|------|------|------|
|criterion|表示使用的计算节点不纯度的方法。|gini entropy|
|random_state|随机数种子，如果固定下来的话，那么最后的结果树以及分数就会固定，通常会随机设置为一个数据，使模型固定以后在对其他参数进行调整。|默认是None，可以填任意整数|
|splitter|best会优先选择重要特征，random会更加随机，深度更深，防止过拟合| 'best','random'|

2. 常用剪枝参数，主要目的是为了防止过拟合

|剪枝参数|说明|使用|
|------|------|------|
|max_depth|限制树的最大深度，防止过拟合|可以从3开始进行调试，主要的分支参数。|
|min_samples_leaf|一个节点在分支后每个节点都至少包含的训练样本数|可以填入浮点数来表示百分比，也可以直接填写想要每个节点至少包含的样本数量|
|min_sample_split|一个节点至少包含这个数值的训练样本数才允许被分支|直接填入数据|
|max_features|分支时涉及到的特征个数。暴力直接进行特征降维|直接填入数据，该参数使用|
|min_impurity_decrease|限制信息增益的大小，即父节点和子节点之间信息熵的差||

3. 重要属性以及接口

1. 属性`clf.feature_importances_`，可以返回每一个特征的重要程度。
2. 属性 `clf.classes_`，返回分类类型
3. 接口`fit，score，apply，predict,`其中fit和score就是模型拟合和打分过程中使用到的接口。`clf.apply(Xtest)`会返回每个测试样本所在叶子节点的索引。`clf.predict(Xtest)`返回Xtest的预测值，可以利用预测值进一步分析。

### 1.3 代码分析

```python
# 表示将wine.data与wine.target连接起来，其中axis是1表示在列上连接。
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1) 
```

```python
# 一般得到数据集时，会手动划分训练集和测试集
# 注意前面的顺序是XXYY，后面是 数据的特征矩阵，数据的标签，划分的百分比
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3) 
# test_size表示百分之三十是测试集，百分之七十是训练集
```

```python
# 绘制决策树
import graphviz
# 可以自定义特征的名字，即原来特征矩阵中的列名
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']

dot_data = tree.export_graphviz(
    clf,# 实例化后的模型
    feature_names=feature_name, # 特征的名字，对用原来的wine.feature_name
    class_names=['五粮液','二锅头','哈尔滨'], # 标签的名字，及分类的结果，对应原来的0,1,2即target
    filled=True, # 表示是否填充颜色
    rounded=True # 表示图形中的方块是圆的
)

graph = graphviz.Source(dot_data)
graph
```

```python
# zip函数将两者打包，然后用*和[]将其解包，那么就是元组的形式存于列表中，然后将其进行排序，最后就是排序后的特征重要性的数据帧
pd.DataFrame([*zip(feature_name,clf.feature_importances_)]).sort_values(1,ascending=0)
```

```python
# 超参数学习曲线，确定最优的剪枝参数，max_depth
import matplotlib.pyplot as plt 
# 创建分数的列表，每打一次分数就将其填入分数的列表中
test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(criterion="entropy"
                                 ,random_state=41
                                 ,splitter='best'
                                 ,max_depth=i+1 # 参数每次循环都改变
                                 )
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    test.append(score)
# 绘制图像
plt.plot(range(1,11),test,color = 'red',label = 'max_depth')
plt.legend()
plt.show()
#从图中可以确定最好的深度
```

### 1.4 决策树回归模型

1. 模型使用

```py
from sklearn.tree import DecisionTreeRegressor
# 实例化
regressor = DecisionTreeRegressor(random_state=0,criterion='squared_error')
# 拟合
regressor.fit(Xtrain,Ytrain)
# 评分
score = clf.score(Xtest, Ytest) #返回预测的准确度
```

2. 模型常用参数及属性

除了criterion的参数值不同之外，其他的参数值都基本相同，含义也基本一致。其中criterion参数的说明如下：

```python
criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"}, default="squared_error"
    The function to measure the quality of a split. Supported criteria are "squared_error" for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, "friedman_mse", which uses mean squared error with Friedman's improvement score for potential splits, "absolute_error" for the mean absolute error, which minimizes the L1 loss using the median of each terminal node, and "poisson" which uses reduction in Poisson deviance to find splits.
```

### 1.5 交叉验证

交叉验证是一种常用的测试模型的方法。交叉验证是用来观察模型的稳定性的一种方法，我们将数据划分为n份，依次使用其中一份作为测试集，其他n-1份作为训练集，多次计算模型的精确性来评估模型的平均准确程度。训练集和测试集的划分会干扰模型的结果，因此用交叉验证n次的结果求出的平均值，是对模型效果的一个更好的度量。

其中交叉验证使用方式：

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
# 导入数据集
boston = load_boston()

# 模型实例化，在这里可以利用循环，改变模型参数进行测试，找出最好的参数取值
regressor = DecisionTreeRegressor(random_state=0)

# 其中cross_val_score 的参数如下 实例化之后的模型 原始的特征的数据集 标签 想要分成的份数 打分的方式
cross_val_score(regressor, boston.data, boston.target, cv=10,
scoring = "neg_mean_squared_error")

# 由于最后得出的结果是一个含有每次分数的列表，在循环中可以将其改为cross_val_score().mean()，以得出平均的数值。
```

### 1.6 网格搜索

1. 优点：通过不同参数的排列组合，得出最优的参数组
2. 耗时

```py
import numpy as np
# gini_threholds = np.linspace(0,0.5,20) # 生成随机的50个有序的值
# entropy_threholds = np.linspace(0,1,20)

#parameters本质上是一串参数，和这一串参数对应的我们希望网格搜索来搜索的参数的取值范围，是一个字典类型

parameters = {
    'criterion':('gini','entropy')
    ,'splitter':('best','random')
    ,'max_depth':[*range(1,10)]
    ,'min_samples_leaf':[*range(1,50,5)]
    # ,'min_impurity_decrease':[*np.linspace(0,0.5,10)]
}

# 模型实例化
clf = DecisionTreeClassifier(random_state=25)
# 网格搜索实例化
Gs = GridSearchCV(clf,parameters,cv=10)
# 进行模型跑分
Gs = Gs.fit(Xtrain,Ytrain)
```

```py
Gs.best_params_ #返回最佳的参数组合
Gs.best_score_ # 返回最好的分数
```





## 2. 随机森林





