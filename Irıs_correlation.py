import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
#We'll use pearson for scale our data to decide if it's significant or not. 
#if P val sets around lower than 0.05 we can consider this data to significant.

iris = pd.read_csv('./iris_data.csv',header=None)
#import data dataset that you want to work with.

iris.columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width','species']

print(iris.head)

print(iris.info)

print(iris.isnull().sum())
#get the summarized data if it contains any "null" data.

print(iris.describe())

iris_corr = iris.select_dtypes(include=['float64','int64'])
#sns heatmap requires to be an 2-D dataset. if your data is 1-Dimensioned so you have to add d-types.

corr, p_value = pearsonr(iris['sepal_length'], iris['petal_length'])
#get correlation and pearson analysis to decide that the data you work with is good to go.

print(f"colleration: {corr}, P-Value:{p_value}") 


corr = iris_corr.corr()

sns.heatmap(corr,annot=True, cmap='coolwarm')
plt.title('Colleration Heatmap')
plt.show()