### Before starting the project:

1. dataset: What kind of problem is this? 

   Are the data continuous/?  If yes then it is regression problrm. (Prediction)

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

