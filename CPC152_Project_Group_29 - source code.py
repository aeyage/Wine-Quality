#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import Library

# In[1]:


# Read Red Wine Dataset by using pandas library
import pandas as pd

# Process Numerical Data
import numpy as np
import numpy.random as npr

# Import Matplotlib and Seaborn for generating graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Clear the warnings
import warnings
warnings.filterwarnings("ignore")


# # Step 2: Import data

# In[2]:


cd C:/Users/behji/Downloads


# In[3]:


# Get data from the dataset
wine = pd.read_csv("winequality.csv") #open file import data


# # Step 3: Understanding data

# In[4]:


# Print the total rows and columns
# Total column that is contained Dataset (1599 rows, 12 columns)
wine.shape 


# In[5]:


wine


# In[6]:


wine.head() #To show first five rows


# In[7]:


wine.tail() #To show last five rows


# In[8]:


wine.describe() #Summary descriptive statistics based solely on numerical numbers


# In[9]:


wine.columns #To check all columns in the file


# In[10]:


wine.dtypes


# In[11]:


# Check the number and types of the columns
wine.info()


# In[12]:


wine.nunique()


# In[13]:


wine['fixed acidity'].unique()


# In[14]:


wine['volatile acidity'].unique()


# In[15]:


wine['citric acid'].unique()


# In[16]:


wine['residual sugar'].unique()


# In[17]:


wine['chlorides'].unique()


# In[18]:


wine['free sulfur dioxide'].unique()


# In[19]:


wine['total sulfur dioxide'].unique()


# In[20]:


wine['density'].unique()


# In[21]:


wine['pH'].unique()


# In[22]:


wine['sulphates'].unique()


# In[23]:


wine['alcohol'].unique()


# In[24]:


wine['quality'].unique()


# In[25]:


# Get actual qualities of the wines
wine['quality'].value_counts()


# In[26]:


wine['quality'].describe()


# # Step 4: Cleaning data

# In[27]:


wine.apply(lambda x: sum(x.isnull()),axis=0)


# # Step 5: Split data

# In[28]:


wine['good wine'] = ['1' if i > 6 else '0' for i in wine['quality']]


# In[29]:


wine['good wine'].value_counts()


# In[30]:


wine


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


train_set, test_set = train_test_split(wine, test_size=0.40,random_state=152)
print(train_set.shape)
print(test_set.shape)


# # Step 6: Features selection

# In[33]:


# Set the size of the figure
plt.subplots(figsize = (10, 10))

# Get the correlation between the features
sns.heatmap(wine[wine.columns].corr(),
            cmap = "Greens",
            annot = True,
            linewidths = 1)

# Set the title
plt.title('Correlation of the features', fontsize = 21)

# Show the correlation graph
plt.show()


# In[34]:


# Show the distribution of qualities of the wines
sns.histplot(data=wine['quality'], binwidth=1, color="violet")
plt.title("Histogram of Qualities of the wines", fontsize = 18)
plt.show()


# # Step 7: Build model

# In[35]:


#import the KNeighborsClasifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier

#import the Decision Tree Classifier class from sklearn
from sklearn.tree import DecisionTreeClassifier

#import the SVC class from sklearn
from sklearn.svm import SVC

# Get the most suitable value of parameter C
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, plot_confusion_matrix


# ## Model 1: KNN

# ### KNN Test 1

# In[36]:


from sklearn.neighbors import KNeighborsClassifier
parameters = {"n_neighbors": range(1,50),
              "weights": ["uniform", "distance"]}
model1gs = GridSearchCV(KNeighborsClassifier(), parameters)
model1gs.fit(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine'])


# In[37]:


model1gs.best_params_


# In[38]:


best_k = model1gs.best_params_["n_neighbors"]
best_weights = model1gs.best_params_["weights"]
bagged_model1 = KNeighborsClassifier(
    n_neighbors=best_k, weights=best_weights
)


# In[39]:


from sklearn.ensemble import BaggingClassifier
bagging_model1 = BaggingClassifier(bagged_model1, n_estimators=100)


# In[40]:


bagging_model1 = KNeighborsClassifier(n_neighbors=23, 
                                      weights='distance',
                                      metric='euclidean'
                                      )


# In[41]:


bagging_model1.fit(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine'])
prediction1_1 = bagging_model1.predict(test_set.drop(['quality', 'good wine'], axis = 1))
prediction1_1


# In[42]:


print("Score the X-train with Y-train is : ", bagging_model1.score(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine']))
print("Score the X-test  with Y-test  is : ", bagging_model1.score(test_set.drop(['quality', 'good wine'], axis = 1), test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'], prediction1_1))
print('Recall: ', recall_score(test_set['good wine'], prediction1_1, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction1_1, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction1_1)
print('Confusion matrix: ')
print(confusion)


# In[43]:


plot_confusion_matrix(bagging_model1, test_set.drop(['quality', 'good wine'], axis = 1), test_set['good wine'])
plt.show()


# ### KNN Test 2

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
parameters = {"n_neighbors": range(1,50),
              "weights": ["uniform", "distance"]}
model1gs2 = GridSearchCV(KNeighborsClassifier(), parameters)
model1gs2.fit(train_set[['citric acid', 'sulphates', 'alcohol']], train_set['good wine'])


# In[45]:


model1gs2.best_params_


# In[46]:


best_k = model1gs2.best_params_["n_neighbors"]
best_weights = model1gs2.best_params_["weights"]
bagged_model1_2 = KNeighborsClassifier(
    n_neighbors=best_k, weights=best_weights
)


# In[47]:


from sklearn.ensemble import BaggingClassifier
bagging_model1_2 = BaggingClassifier(bagged_model1_2, n_estimators=100)


# In[48]:


bagging_model1_2 = KNeighborsClassifier(
    n_neighbors=25,
    weights="distance",
    metric="euclidean"
)


# In[49]:


bagging_model1_2.fit(train_set[['citric acid', 'sulphates', 'alcohol']], train_set['good wine'])
prediction1_2 = bagging_model1_2.predict(test_set[['citric acid', 'sulphates', 'alcohol']])
prediction1_2


# In[50]:


print("Score the X-train with Y-train is : ", bagging_model1_2.score(train_set[['citric acid','sulphates','alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", bagging_model1_2.score(test_set[['citric acid','sulphates', 'alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction1_2))
print('Recall: ', recall_score(test_set['good wine'], prediction1_2, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction1_2, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction1_2)
print('Confusion matrix: ')
print(confusion)


# In[51]:


plot_confusion_matrix(bagging_model1_2, test_set[['citric acid','sulphates','alcohol']], test_set['good wine'])
plt.show()


# ### KNN Test 3

# In[52]:


from sklearn.neighbors import KNeighborsClassifier
parameters = {"n_neighbors": range(1,50),
              "weights": ["uniform", "distance"]}
model1gs3 = GridSearchCV(KNeighborsClassifier(), parameters)
model1gs3.fit(train_set[['sulphates', 'alcohol']], train_set['good wine'])


# In[53]:


model1gs3.best_params_


# In[54]:


best_k = model1gs3.best_params_["n_neighbors"]
best_weights = model1gs3.best_params_["weights"]
bagged_model1_3 = KNeighborsClassifier(
    n_neighbors=best_k, weights=best_weights
)


# In[55]:


from sklearn.ensemble import BaggingClassifier
bagging_model1_3 = BaggingClassifier(bagged_model1_3, n_estimators=100)


# In[56]:


bagging_model1_3 = KNeighborsClassifier(
    n_neighbors=29,
    weights="uniform",
    metric="euclidean"
)


# In[57]:


bagging_model1_3.fit(train_set[['sulphates', 'alcohol']], train_set['good wine'])
prediction1_3 = bagging_model1_3.predict(test_set[['sulphates', 'alcohol']])
prediction1_3


# In[58]:


print("Score the X-train with Y-train is : ", bagging_model1_3.score(train_set[['sulphates','alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", bagging_model1_3.score(test_set[['sulphates', 'alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction1_3))
print('Recall: ', recall_score(test_set['good wine'], prediction1_3, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction1_3, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction1_3)
print('Confusion matrix: ')
print(confusion)


# In[59]:


plot_confusion_matrix(bagging_model1_3, test_set[['sulphates','alcohol']], test_set['good wine'])
plt.show()


# ### KNN Test 4

# In[60]:


from sklearn.neighbors import KNeighborsClassifier
parameters = {"n_neighbors": range(1,50),
              "weights": ["uniform", "distance"]}
model1gs4 = GridSearchCV(KNeighborsClassifier(), parameters)
model1gs4.fit(train_set[['alcohol']], train_set['good wine'])


# In[61]:


model1gs4.best_params_


# In[62]:


best_k = model1gs4.best_params_["n_neighbors"]
best_weights = model1gs4.best_params_["weights"]
bagged_model1_4 = KNeighborsClassifier(
    n_neighbors=best_k, weights=best_weights
)


# In[63]:


bagging_model1_4 = KNeighborsClassifier(
    n_neighbors=31,
    weights="uniform",
    metric="euclidean"
)


# In[64]:


bagging_model1_4.fit(train_set[['alcohol']], train_set['good wine'])
prediction1_4 = bagging_model1_4.predict(test_set[['alcohol']])
prediction1_4


# In[65]:


print("Score the X-train with Y-train is : ", bagging_model1_4.score(train_set[['alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", bagging_model1_4.score(test_set[['alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction1_4))
print('Recall: ', recall_score(test_set['good wine'], prediction1_4, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction1_4, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction1_4)
print('Confusion matrix: ')
print(confusion)


# In[66]:


plot_confusion_matrix(bagging_model1_4, test_set[['alcohol']], test_set['good wine'])
plt.show()


# ## Model 2: Decision Tree
# ### Decision Tree Test 1

# In[67]:


model2 = DecisionTreeClassifier(random_state = 152)


# In[68]:


parameter = {"max_depth": range(1,6), "max_features": range(1,10),
              "criterion": ["gini", "entropy"]}
model2_cv = GridSearchCV(model2, parameter, cv=5)
model2_cv.fit(train_set.drop(['quality', 'good wine'], axis = 1),train_set['good wine'])
print(model2_cv.best_params_)


# In[69]:


model2 = DecisionTreeClassifier(random_state = 152,
                                criterion = 'entropy',
                                max_depth = 5,
                                max_features = 4)


# In[70]:


model2.fit(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine'])
prediction2_1 = model2.predict(test_set.drop(['quality', 'good wine'], axis = 1))
prediction2_1


# In[71]:


print("Score the X-train with Y-train is : ", model2.score(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model2.score(test_set.drop(['quality', 'good wine'], axis = 1), test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'], prediction2_1))
print('Recall: ', recall_score(test_set['good wine'], prediction2_1, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction2_1, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction2_1)
print('Confusion matrix: ')
print(confusion)


# In[72]:


plot_confusion_matrix(model2, test_set.drop(['quality', 'good wine'], axis = 1), test_set['good wine'])
plt.show()


# ### Decision Tree Test 2

# In[73]:


model2 = DecisionTreeClassifier(random_state=152)


# In[74]:


parameter = {"max_depth": range(1,6), "max_features": range(1,10),
              "criterion": ["gini", "entropy"]}
model2_cv = GridSearchCV(model2, parameter, cv=5)
model2_cv.fit(train_set[['citric acid', 'sulphates', 'alcohol']],train_set['good wine'])
print(model2_cv.best_params_)


# In[75]:


model2 = DecisionTreeClassifier(random_state=152, criterion='entropy', max_depth = 3, max_features=2)


# In[76]:


model2.fit(train_set[['citric acid', 'sulphates', 'alcohol']], train_set['good wine'])
prediction2_2 = model2.predict(test_set[['citric acid', 'sulphates', 'alcohol']])
prediction2_2


# In[77]:


print("Score the X-train with Y-train is : ", model2.score(train_set[['citric acid','sulphates','alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model2.score(test_set[['citric acid','sulphates', 'alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction2_2))
print('Recall: ', recall_score(test_set['good wine'], prediction2_2, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction2_2, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction2_2)
print('Confusion matrix: ')
print(confusion)


# In[78]:


plot_confusion_matrix(model2, test_set[['citric acid','sulphates','alcohol']], test_set['good wine'])
plt.show()


# ### Decision Tree Test 3

# In[79]:


model2 = DecisionTreeClassifier(random_state=152)


# In[80]:


parameter = {"max_depth": range(1,6), "max_features": range(1,10),
              "criterion": ["gini", "entropy"]}
model2_cv = GridSearchCV(model2, parameter, cv=5)
model2_cv.fit(train_set[['sulphates', 'alcohol']], train_set['good wine'])
print(model2_cv.best_params_)


# In[81]:


model2 = DecisionTreeClassifier(random_state=152, criterion='gini', max_depth = 3, max_features=2)


# In[82]:


model2.fit(train_set[['sulphates', 'alcohol']], train_set['good wine'])
prediction2_3 = model2.predict(test_set[['sulphates', 'alcohol']])
prediction2_3


# In[83]:


print("Score the X-train with Y-train is : ", model2.score(train_set[['sulphates','alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model2.score(test_set[['sulphates', 'alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction2_3))
print('Recall: ', recall_score(test_set['good wine'], prediction2_3, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction2_3, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction2_3)
print('Confusion matrix: ')
print(confusion)


# In[84]:


plot_confusion_matrix(model2, test_set[['sulphates','alcohol']], test_set['good wine'])
plt.show()


# ### Decision Tree Test 4

# In[85]:


model2 = DecisionTreeClassifier(random_state=152)


# In[86]:


parameter = {"max_depth": range(1,6), "max_features": range(1,10),
              "criterion": ["gini", "entropy"]}
model2_cv = GridSearchCV(model2, parameter, cv=5)
model2_cv.fit(train_set[['alcohol']], train_set['good wine'])
print(model2_cv.best_params_)


# In[87]:


model2 = DecisionTreeClassifier(random_state=152, criterion='gini', max_depth = 1, max_features=1)


# In[88]:


model2.fit(train_set[['alcohol']], train_set['good wine'])
prediction2_4 = model2.predict(test_set[['alcohol']])
prediction2_4


# In[89]:


print("Score the X-train with Y-train is : ", model2.score(train_set[['alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model2.score(test_set[['alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction2_4))
print('Recall: ', recall_score(test_set['good wine'], prediction2_4, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction2_4, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction2_4)
print('Confusion matrix: ')
print(confusion)


# In[90]:


plot_confusion_matrix(model2, test_set[['alcohol']], test_set['good wine'])
plt.show()


# ## Model 3: SVM

# ### SVM Test 1

# In[91]:


# Create Model 3 for SVC
model3 = SVC(random_state = 152,
             gamma = 'auto')


# In[92]:


# Find the most suitable C
grid = {'C': [1, 1.1, 1.2, 1.3, 1.4, 2]}

model3_cv = GridSearchCV(model3, grid, cv = 5)
model3_cv.fit(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine'])

# Show the result
model3_cv.best_estimator_


# In[93]:


# Apply the change of C
model3 = SVC(C = 1.1,
             random_state = 152,
             gamma = 'auto')


# In[94]:


model3.fit(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine'])
prediction3_1 = model3.predict(test_set.drop(['quality', 'good wine'], axis = 1))
prediction3_1


# In[95]:


print("Score the X-train with Y-train is : ", model3.score(train_set.drop(['quality', 'good wine'], axis = 1), train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model3.score(test_set.drop(['quality', 'good wine'], axis = 1), test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'], prediction3_1))
print('Recall: ', recall_score(test_set['good wine'], prediction3_1, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction3_1, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction3_1)
print('Confusion matrix: ')
print(confusion)


# In[96]:


plot_confusion_matrix(model3, test_set.drop(['quality', 'good wine'], axis = 1), test_set['good wine'])
plt.show()


# ### SVM Test 2

# In[97]:


model3 = SVC(random_state = 152,
             gamma = 'auto')


# In[98]:


# Find the most suitable C
grid = {'C': [0.9, 1, 1.1, 1.5, 1.9, 2, 2.1, 2.3, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]}

model3_cv = GridSearchCV(model3, grid, cv = 5)
model3_cv.fit(train_set[['citric acid', 'sulphates', 'alcohol']], train_set['good wine'])

# Show the result
model3_cv.best_estimator_


# In[99]:


# Apply the change of C
model3 = SVC(C = 5,
             random_state = 152,
             gamma = 'auto')


# In[100]:


model3.fit(train_set[['citric acid', 'sulphates', 'alcohol']], train_set['good wine'])
prediction3_2 = model3.predict(test_set[['citric acid', 'sulphates', 'alcohol']])
prediction3_2


# In[101]:


print("Score the X-train with Y-train is : ", model3.score(train_set[['citric acid','sulphates','alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model3.score(test_set[['citric acid','sulphates', 'alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction3_2))
print('Recall: ', recall_score(test_set['good wine'], prediction3_2, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction3_2, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction3_2)
print('Confusion matrix: ')
print(confusion)


# In[102]:


plot_confusion_matrix(model3, test_set[['citric acid','sulphates','alcohol']], test_set['good wine'])
plt.show()


# ### SVM Test 3

# In[103]:


model3 = SVC(random_state = 152,
             gamma = 'auto')


# In[104]:


# Find the most suitable C
grid = {'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]}

model3_cv = GridSearchCV(model3, grid, cv = 5)
model3_cv.fit(train_set[['sulphates', 'alcohol']], train_set['good wine'])

# Show the result
model3_cv.best_estimator_


# In[105]:


# Apply the change of C
model3 = SVC(C = 5,
             random_state = 152,
             gamma = 'auto')


# In[106]:


model3.fit(train_set[['sulphates', 'alcohol']], train_set['good wine'])
prediction3_3 = model3.predict(test_set[['sulphates', 'alcohol']])
prediction3_3


# In[107]:


print("Score the X-train with Y-train is : ", model3.score(train_set[['sulphates','alcohol']],train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model3.score(test_set[['sulphates', 'alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'], prediction3_3))
print('Recall: ', recall_score(test_set['good wine'], prediction3_3, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction3_3, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction3_3)
print('Confusion matrix: ')
print(confusion)


# In[108]:


plot_confusion_matrix(model3, test_set[['sulphates','alcohol']], test_set['good wine'])
plt.show()


# ### SVM Test 4

# In[109]:


model3 = SVC(random_state = 152,
             gamma = 'auto')


# In[110]:


# Find the most suitable C
grid = {'C': [0.0001, 0.00011, 0.1, 0.5, 1]}

model3_cv = GridSearchCV(model3, grid, cv = 5)
model3_cv.fit(train_set[['alcohol']], train_set['good wine'])

# Show the result
model3_cv.best_estimator_


# In[111]:


# Apply the change of C
model3 = SVC(C = 0.0001,
             random_state = 152,
             gamma = 'auto')


# In[112]:


model3.fit(train_set[['alcohol']], train_set['good wine'])
prediction3_4 = model3.predict(test_set[['alcohol']])
prediction3_4


# In[113]:


print("Score the X-train with Y-train is : ", model3.score(train_set[['alcohol']], train_set['good wine']))
print("Score the X-test  with Y-test  is : ", model3.score(test_set[['alcohol']], test_set['good wine']))
print()
print('Accuracy: ', accuracy_score(test_set['good wine'],prediction3_4))
print('Recall: ', recall_score(test_set['good wine'], prediction3_4, average="weighted"))
print('Precision: ', precision_score(test_set['good wine'], prediction3_4, average="weighted"))
print()
confusion = confusion_matrix(test_set['good wine'], prediction3_4)
print('Confusion matrix: ')
print(confusion)


# In[114]:


plot_confusion_matrix(model3, test_set[['alcohol']], test_set['good wine'])
plt.show()

