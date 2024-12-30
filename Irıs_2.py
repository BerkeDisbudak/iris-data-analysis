import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

iris = pd.read_csv('iris_data.csv')

iris.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']

print(iris.columns)
print(iris.isnull().sum())
"""
Note: This code snippet were for EDA.

iris['Species'].value_counts().plot(kind='bar', color =['skyblue','orange','red'])
plt.title('Species counts.')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

sns.scatterplot(data = iris, x ='Sepal_Length', y='Petal_Length', hue='Species')
plt.title('Petal vs Sepal lengths')
plt.show()

sns.scatterplot(data = iris, x = 'Sepal_Length', y = 'Sepal_Width', hue= 'Species')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

sns.boxplot(data = iris, x='Species', y='Sepal_Length')
plt.title('Sepal Length Distribution by Species.')
plt.xlabel('Species')
plt.ylabel('Sepal_Length')
plt.show()
"""

#KNN methodology starts here, do not forget to import crucial libraries for it.
scaler = StandardScaler()
#StandardScaler only works with numerical data, do not forget. so do not ever inclue 'Species' and double check with .dtypes.
features = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
iris[features] = scaler.fit_transform(iris[features])

print(iris.head)
print(iris[features].head())

iris_scaled = iris.copy()
iris_scaled[features] = scaler.fit_transform(iris[features])
print(iris_scaled.head())

#For splitting those dataset to train model with X and Y shapes(Which is Train set and learning sets.) in order to create an KNN's.
X = iris[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
Y = iris['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)
print('Accuracy:', accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(knn, X_test, Y_test, cmap='Blues')
plt.title('Confuson Matrix.')
plt.show()


