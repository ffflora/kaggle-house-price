### Before starting the project:

1. dataset: What kind of problem is this? 

   Are the data continuous/?  If yes then it is regression problem. (Prediction)

2. Think what models could be using? (linear regression)
3. What kind of data needed to fit linear regression?
4. How to deal with missing data?
5. EDA (**pandas_profiling**), feature selection, feature engineer
6. Model selection



### Data Cleaning:

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

### Feature Engineering

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

  

  

### Submission:

- v1: 12.91779

  Label encoding + get_dummies, model: linear Regression 
  
- v2: 0.21173

  Feature Engineering