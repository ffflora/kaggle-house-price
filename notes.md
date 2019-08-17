## Before starting the project:

1. dataset: What kind of problem is this? 

   Are the data continuous/?  If yes then it is regression problem. (Prediction)

2. Think what models could be using? (linear regression)
3. What kind of data needed to fit linear regression?
4. How to deal with missing data?
5. EDA (**pandas_profiling**), feature selection, feature engineer
6. Model selection



## Data Cleaning:

1. Missing value:  use mean, max, min to replace NAN

2. remove duplicate

3. remove invalid value (outliers), use correlation analysis 识别不遵守分布或回归方程的值，也可以用简单规则库（ 常识性规则、业务待定规则等）检查数据值，或使用不同属性间的约束、外部的数据来检测和清理数据。

4. 解决数据的不一致性：on-hot encode 

   

### Scenes of Data Cleaning 

1. drop columns
2. change data types 
3. categorical -> numerical 
4. check missing values
5. check strings in the column
6. delete empty spaces in the cols. `dropna`
7. feature cross
8. string -> timestamp

## Feature Engineering

- 不属于**同一量纲**：特征的规格不一样，不能放在一起比较。 

  - `StandardScaler` 可以解决这个问题 (z-score standarization) `x' = (x - xbar)/S`

    ```python
    from sklearn.preprocessing import StandardScaler
    StandardScaler().fit_transform(...data...)
    ```

  - `MinMaxScaler` 区间缩放，对列向量处理，[0,1]归一化

    ```python
    from sklearn.preprocessing import MinMaxScaler
    MinmMaxScaler().fit_transform(...data...)
    ```

  - Which one is better?

    在后续的分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA、LDA这些需要用到协方差分析进行降维的时候， 同时数据分布可以近似为normal distribution 的时候，z-score standardization is better.

    在不涉及距离度量、协方差计算、数据不符合正态分布的时候，`MinMaxScaler` is better. 比如图像处理中，RGB图像转换为灰度图像后将其值限定在[0,255]的区间。

  - 对行向量进行处理：`normalizer`

    ```python
    from sklearn.preprocessing import Normalizer
    norm = Normalizer()
    norm.fit_transform(...data...)
    ```

    

- 信息冗余：【及格】【不及格】-> 【0】【1】

  - 对列向量进行处理

    **定性与定量区别** 
    定性：博主很胖，博主很瘦 
    定量：博主有80kg，博主有60kg 
    一般定性都会有相关的描述词，定量的描述都是可以用数字来量化处理 

    定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
    `Binarizer`

    ```python
    from sklearn.preprocessing import Binarizer
    
    #二值化，阈值设置为3，返回值为二值化后的数据
    Binarizer(threshold=3).fit_transform(iris.data) 
    # >3 -> 1; <3 -> 0
    ```

- 定性特征不能直接使用:  -> `one-hot`

  - 对列向量进行处理

    因为有些特征是用文字分类表达的，或者说将这些类转化为数字，但是数字与数字之间是没有大小关系的，纯粹的分类标记，这时候就需要用哑编码对其进行编码。`iris` 数据集的特征皆为定量特征，使用其目标值进行哑编码（实际上是不需要的）。

    ```python
    from sklearn.preprocessing import OneHotEncoder
    
    #哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
     OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))
    ```

- missing value: remove/fill na

  ```python
  1 from numpy import vstack, array, nan
  2 from sklearn.preprocessing import Imputer
  3 
  4 #缺失值计算，返回值为计算缺失值后的数据
  5 #参数missing_value为缺失值的表示形式，默认为NaN
  6 #参数strategy为缺失值填充方式，默认为mean（均值）
  7 Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
  # 4个特征均赋值为NaN，表示数据缺失。
  ```

  


## Ensemble Learning

- Bagging
  - Random Forest

- Boosting
  - AdaBoost
  - GBDT
  - XGBoost
  - LightGBM
  - CatBoost

- Stacking

- Blending

```python
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3 = LogisticRegression()

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
```



#### Maximum Voting method 最大投票法

通常用于分类问题，使用多种模型预测每个数据，每个模型的预测都被视为一次投票。大多数模型得到的预测被用于最终预测结果。

```python
pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)
pred3 = model3.predict(x_test)

final = np.array([])
for i in range(0,len(x_test)):
    final = np.append(final, mode([pred1[i],pred2[i],pred3[i]]))
```



#### Average method 平均法

类似于最大投票法，对每个数据多次预测进行平均。平均法可用于在回归问题中进行预测或在计算分类问题的概率时使用。

```python
pred1 = model1.predict_proba(x_test)
pred2 = model2.predict_proba(x_test)
pred3 = model3.predict_proba(x_test)

final = (pred1 + pred2 + pred3)/3
```



#### Weighted average method 加权平均法

```python
pred1 = model1.predict_proba(x_test)
pred2 = model2.predict_proba(x_test)
pred3 = model3.predict_proba(x_test)

final = (pred1 *0.3 + pred2*0.4 + pred3*0.3)/(0.3 + 0.4 + 0.3)
```



### 高级集成技术

1. 模型的堆叠 stacking
2. 模型的融合 blending

#### stacking: 

1. divide training set into 10 pieces
2. fit one of the weak model on 9 of them, and predict the 10th
3. repeat (predict) it one every piece of data 
4. fit the whole data set
5. predict the test set
6. repeat the 2-4th step with other weak models 
7. the prediction of the training set are considered as the feature of the new model

```python
# n-fold valifation 
# 此函数返回每个模型对训练集和测试集的预测

def stacking(model,train,y,test,n_fold):
	folds = StratifiedKFold(n_splits = n_fold,random_state = 1)
    test_pred = np.empty((test.shape[0],1),float)
    train_pred = np.empty((0,1),float)
    for train_indx,val_indx in folds.split(train,y.values):
        x_train,x_val = train.iloc[train_indx],train.iloc[val_indx]
        y_train,y_val = y.iloc[train_indx],y.iloc[val_indx]
        
        model.fit(X=x_train,y=y_train)
        train_pred = np.append(train_pred,model.predict(x_val)) # using 1/n of the training set to get the 'input' of the next layer
        test_pred = np.append(test_pred,model.predict(test)) # 得到的是这个 test set 的新特征
  	return test_pred.reshape(-1,1),train_pred
    
```

现在创建两个基本模型：Decision Tree and KNN

```python
model1= tree.DecisionTreeClassifier(random_state = 1)

test_pred1, train_pred1 = Stacking(model = model1,n_fold = 10, train= x_train, test + x_trest, y= y_train)
train_pred1 = pd.DataFrame(train_pred1)
test_pred1 = pd.DataFrame(test_pred1)

model2= KNeighborsClassifier()

test_pred2, train_pred2 = Stacking(model = model2,n_fold = 10, train= x_train, test = x_trest, y= y_train)
train_pred2 = pd.DataFrame(train_pred2)
test_pred2 = pd.DataFrame(test_pred2)

```

创建第三个模型，逻辑回归，在 决策树和 KNN 的预测之上。

```python
df = pd.concat([train_pred1, train_pred2],axis = 1)
df_test = pd.concat([test_pred1,test_pred2],axis = 1)

model = LogisticRegression(random_state = 1)
model.fit(df,y_train)
model.score(df_score,y_test)
```

Decision Tree and KNN 在第0层，LR 在第一层。



## Submission:

- v1: 12.91779

  Label encoding + get_dummies, model: linear Regression 
  
- v2: 0.21173

  Feature Engineering

- v3: 0.12563

  model ensemble