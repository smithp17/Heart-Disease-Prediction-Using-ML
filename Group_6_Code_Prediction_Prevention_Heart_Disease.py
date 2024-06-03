#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
color = sns.color_palette()
from IPython.display import display
pd.options.display.max_columns = None
import keras
import pickle

# Common sklearn Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# sklearn modules for performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score



from sklearn.model_selection import train_test_split  # import 'train_test_split'
from sklearn.metrics import confusion_matrix, average_precision_score, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[4]:


data_frame = pd.read_csv("C:/Users/16674/Desktop/UMBC-MS_CS/First_Sem/Introduction to Machine Learning/Project/framingham.csv")


# In[5]:


data_frame.head()


# In[6]:


#Shape of dataset

print ('No. of Records :', data_frame.shape[0], '\nNo. of Features : ', data_frame.shape[1])


# In[7]:


#Let us check datset's attribute info

data_frame.info()


# In[8]:


data_frame.describe()


# Observations :
# 
# 1. Some the features are Discrete so let us analyze continuous ones
# 2. Age : We can see that Min. age of subject found in given records is 32 while Max. being 70. So our values are ranging from 32 to 70.
# 3. cigsPerDay : Subject smoking Cig. per day is as low as nill while we have 70 Cigs. per day making the Peak.
# 4. totChol : Min. Cholesterol level recorded in our dataset is 107 while Max. is 696.
# 5. sysBP : Min. Systolic Blood Pressure observed in Subject is 83 while Max. is 295.
# 6. diaBP : Min. Diastolic Blood Pressure observed in Subject is 48 while Max. is 142.
# 7. BMI : Body Mass Index in our dataset ranges from 15.54 to 56.
# 8. heartRate : Observed Heartrate in our case study is 44 to 143.
# 9. glucose : Glucose sugar level range is 40 to 394.

# In[9]:


#Examining Null values in each feature

data_frame.isnull().sum()


# ##  Handle missing data from the dataset by using median

# In[10]:


#User defined function for missing value imputation

def impute_median(data):
    return data.fillna(data.median())


# In[11]:


#median imputation

data_frame.glucose = data_frame['glucose'].transform(impute_median)
data_frame.education = data_frame['education'].transform(impute_median)
data_frame.heartRate = data_frame['heartRate'].transform(impute_median)
data_frame.totChol = data_frame['totChol'].transform(impute_median)
data_frame.BPMeds = data_frame['BPMeds'].transform(impute_median)

## group by classes that are in relation with other classes

by_currentSmoker = data_frame.groupby(['currentSmoker'])
data_frame.cigsPerDay = by_currentSmoker['cigsPerDay'].transform(impute_median)

by_age = data_frame.groupby(['male','age'])
data_frame.BMI = by_age['BMI'].transform(impute_median)


# In[12]:


data_frame = data_frame.dropna(how = 'any', axis = 0)
print(data_frame.shape)
data_frame.head()


# In[13]:


#Rechecking if we have any missing value left

data_frame.isnull().sum()


# In[14]:


#Shape of dataset

print ('No. of Records :', data_frame.shape[0], '\nNo. of Features : ', data_frame.shape[1])


# ## Descriptive Statistics

# In[15]:


print('Gender')
print(data_frame['male'].value_counts(normalize = True))
print('----')
print('\n')

print('Education')
print(data_frame['education'].value_counts(normalize = True))
print('----')
print('\n')

print('BP Medication')
print(data_frame['BPMeds'].value_counts(normalize = True))
print('----')
print('\n')

print('Stroke')
print(data_frame['prevalentStroke'].value_counts(normalize = True))
print('----')
print('\n')

print('Hypertension')
print(data_frame['prevalentHyp'].value_counts(normalize = True))
print('----')
print('\n')

print('Diabetes')
print(data_frame['diabetes'].value_counts(normalize = True))
print('----')
print('\n')


# ## Conclusion drawn from above operation-
# 1. Assuming that 0 is female and 1 is male - 57% is Female, 42% is Male
# 2. Most of the patients in the database (60%+) is below education level 2 and lower
# 3. 97% of the patients are not on BP Medication
# 4. 99% have not had a Stroke before
# 5. About 69% are not Hypertension patients and 31% are
# 6. 97% users are not diabetic

# In[16]:


disease = data_frame.groupby('TenYearCHD')

print('Gender')
print(disease['male'].value_counts(normalize = True))
print('----')
print('\n')

print('Education')
print(disease['education'].value_counts(normalize = True))
print('----')
print('\n')

print('BP Medication')
print(disease['BPMeds'].value_counts(normalize = True))
print('----')
print('\n')

print('Stroke')
print(disease['prevalentStroke'].value_counts(normalize = True))
print('----')
print('\n')

print('Hypertension')
print(disease['prevalentHyp'].value_counts(normalize = True))
print('----')
print('\n')

print('Diabetes')
print(disease['diabetes'].value_counts(normalize = True))
print('----')
print('\n')


# ## Conclusion-
# 1. Males seem to be slightly susceptible to Heart Disease compared to Females. (Basicaly Gender may play a role)
# 2. While it looks like lower education level patients are more susceptible, the overall number of lower education level patients are also much higher
# 3. BP Medication, Stroke and Diabetes dont seem to have too much of an impact
# 4. Hypertension, however, seems like it has an impact.

# In[17]:


data_frame['totChol'].max()


# In[18]:


data_frame['sysBP'].max()


# In[19]:


data_frame = data_frame[data_frame['totChol']<600.0]
data_frame = data_frame[data_frame['sysBP']<295.0]
data_frame.shape


# ## Exploratory Data Analysis

# In[20]:


sns.set_context('talk')
plt.figure(figsize=(22,10))
data_heatmap = data_frame[['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']]
sns.heatmap(data_heatmap.corr()*100, annot=True, cmap='BrBG')


# There seems to be a decently strong correlation between sysBP and diaBP as well as there is negative corelation between education and output variable. Hence we will remove this column later

# In[21]:


sns.set_context('talk')
plt.figure(figsize=(22,10))
sns.heatmap(data_frame.corr()*100, annot=True, cmap='BrBG')


# In[22]:


categorical_features = ['male','currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
for feature in categorical_features:
    print(feature,':')
    print(data_frame[feature].value_counts())
    print("-----------------")


# In[23]:


num_plots = len(categorical_features)
total_cols = 4
total_rows = num_plots//total_cols + 1
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(7*total_cols, 7*total_rows), facecolor='w', constrained_layout=True)
for i, var in enumerate(categorical_features):
    row = i//total_cols
    pos = i % total_cols
    plot = sns.countplot(x=var, data=data_frame, ax=axs[row][pos])


# ## Among the categorical features:
# 
# BPmeds, prevalentStroke and diabetes are highly imbalanced.
# There are four levels of education whereas the rest categorical features are all binary
# The number of Smokers and non-Smokers in currentSmoker is almost the same

# In[24]:


#CHD by Gender Viz.
sns.catplot(x='male', hue='TenYearCHD', data=data_frame, kind='count', palette='Dark2', legend=False)
plt.xlabel('Gender')
plt.xticks(ticks=[0,1], labels=['Female', 'Male'])
plt.ylabel('No. of Patients')
plt.legend(['Neg.', 'Pos.'])
plt.title('CHD by Gender')


# Observations :
# 
# 1. Above Bivariate Analysis plot depicts Gender wise absence / presence of Chronic Heart Disease (CHD).
# 2. Observations tells us that we've Excessive number of people who are not suffering from CHD.
# 3. Negative : Approx. 80 to 90% of Females are falling in Negative Category while Approx. 60 to 70% of Males are in Negative Slot.
# 4. Positive : While Approx. 10% of Females & Males are suffering from CHD.
# 5. By this we can say that our Dataset is Imbalanced where Approx. 80 to 90% are Negative Classifications and Approx. 10 to 15% are Positive Classes.

# In[25]:


# checking distributions using histograms
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data_frame.hist(ax = ax)


# In[26]:


#Distribution of Continuous variables

plt.figure(figsize=(23,15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(2, 3, 1)
sns.distplot(data_frame['glucose'] , color='orange')
plt.title('Distribution of Glucose')

plt.subplot(2, 3, 2)
sns.distplot(data_frame['totChol'], color='g')
plt.title('Distribution of Total Cholesterol')

plt.subplot(2, 3, 3)
sns.distplot(data_frame['sysBP'], color='r')
plt.title('Distribution of Systolic BP')

plt.subplot(2, 3, 4)
sns.distplot(data_frame['diaBP'] , color='purple')
plt.title('Distribution of Dia. BP')

plt.subplot(2, 3, 5)
sns.distplot(data_frame['BMI'], color='blue')
plt.title('Distribution of BMI')

plt.subplot(2, 3, 6)
sns.distplot(data_frame['heartRate'], color='grey')
plt.title('Distribution of HeartRate')


# Observations :
# 
# 1. We can see Glucose, Total Cholesterol, Systolic BP & BMI is Right Skewed.
# 2. While Diastolic BP & Heart Rate are close to Normal / Gaussian Distribution

# In[27]:


sns.distplot(data_frame['age'], bins=15, kde=True, color='maroon')
plt.ylabel('Count')
plt.title('Agewise distribution of the patients')


# In[28]:


dataset_binary = ['male', 'currentSmoker','BPMeds','prevalentHyp', 'diabetes']

fig, ax = plt.subplots (len(dataset_binary), figsize = (18, 20))
for n,k in enumerate(dataset_binary):
    sns.countplot(x=data_frame['TenYearCHD'], hue = data_frame[k], ax=ax[n])
    ax[n].set_title(" A boxplot representing the column" + " " + k, fontsize = 15)
    fig.tight_layout(pad = 1.1)


# ## The above plots show that male and Current Smokers are at high risk of getting a heart disease

# In[29]:


dataset_num = ['age','cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
fig, ax = plt.subplots (len(dataset_num), figsize = (18, 23))
for n,k in enumerate(dataset_num):
    sns.barplot(x=data_frame['TenYearCHD'], y = data_frame[k], ax=ax[n])
    ax[n].set_title(" A boxplot representing the column" + " " + k, fontsize = 15)
    fig.tight_layout(pad = 1.1)


# Observation :
# 
# Subjects ranging from Age 40 to 50 are in Majority followed by 50 to 70.

# In[30]:


#User-defined function

#Age encoding
def encode_age(data):
    if data <= 40:
        return 0
    if data > 40 and data <=55:
        return 1
    else:
        return 2    

#heart rate encoder
def heartrate_enc(data):
    if data <= 60:
        return 0
    if data > 60 and data <=100:
        return 1
    else:
        return 2

#applying functions
data_frame['enc_hr'] = data_frame['heartRate'].apply(heartrate_enc)
data_frame['encode_age'] = data_frame['age'].apply(lambda x : encode_age(x))


# In[31]:


plt.figure(figsize=(23,8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.subplot(1, 2, 1)
sns.boxenplot(x='encode_age', y='sysBP', hue='male', data=data_frame, palette='rainbow')
plt.xlabel('Age Group / Gender')
plt.ylabel('Sys. BP')
plt.xticks(ticks=[0,1,2], labels=['Adults', 'Middle-Aged', 'Senior'])
plt.title('Sys. BP by Age Group & Gender')
plt.legend(title='Gender')

plt.subplot(1, 2, 2)
sns.boxenplot(x='encode_age', y='diaBP', hue='male', data=data_frame, palette='pastel')
plt.xlabel('Age Group / Gender')
plt.ylabel('Dia. BP')
plt.xticks(ticks=[0,1,2], labels=['Adults', 'Middle-Aged', 'Senior'])
plt.title('Dia. BP Count by Age Group')
plt.legend(title='Gender')


# Observations :
# 
# 1. Sys. BP by Age Group & Gender : Sys. BP is Increasing by Age Group and Gender.
# 2. Dia. BP by Age Group & Gender : Similar to Sys. BP , the Dia. BP is seen Increasing by Age Group & Gender.

# In[32]:


#Multivariate Analysis Pt. 1

plt.figure(figsize=(23,8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.subplot(1, 2, 1)
sns.boxenplot(x='encode_age', y='glucose', hue='male', data=data_frame, palette='seismic')
plt.xlabel('Age Group / Gender')
plt.ylabel('Glucose')
plt.xticks(ticks=[0,1,2], labels=['Adults', 'Middle-Aged', 'Senior'])
plt.title('Glucose Count by Age Group & Gender')
plt.legend(title='Gender')

plt.subplot(1, 2, 2)
sns.boxenplot(x='encode_age', y='totChol', hue='male', data=data_frame, palette='Accent')
plt.xlabel('Age Group / Gender')
plt.ylabel('Total Cholesterol')
plt.xticks(ticks=[0,1,2], labels=['Adults', 'Middle-Aged', 'Senior'])
plt.title('Total Chol. Count by Age Group')
plt.legend(title='Gender')


# Observations :
# 
# 1. Glucose Count by Age Group & Gender : We can clearly observe that as Age increases the count of Glucose increases too. While Gender wise Glucose Count has almost similiar Median with Few outliers in each.
# 2. Total Cholesterol by Age Group & Gender : Excluding Outliers, Observation make us Clear that for females Cholesterol level is Increasing by Age considering the Quantile (25%, 50%, 75%) values into account. While, for Males the Cholesterol level Quantile is Approx. Similar for each Age Group.

# In[33]:


#Violin Plot of Cigsperday by age group

sns.catplot(data=data_frame, x='encode_age', y='cigsPerDay', kind='violin', size=7, palette='Greys_r')
plt.xlabel('Age Group / Gender')
plt.ylabel('Cigs. / Day')
plt.xticks(ticks=[0,1,2], labels=['Adults', 'Middle-Aged', 'Senior'])
plt.title('Cigs. per day by Age Group')


# ## Observation :
# 
# 1. Adults : In Adults we can observe that Median values has Lower Kernel Density followed by 75% IQR's Density. While, 25% IQR marks the Higher Kernel Density.
# 2. Middle-Aged : In Middle-Aged Group we can observe that 25% IQR & Median has Higher Kernel Density while 75% IQR has a quite Lower Kernel Density.
# 3. Senior : In Seniority section we can observe that Median and 25% IQR are Closely Intact to each other having Higher Kernel Density, while 75% IQR got Lower Kernel Density.

# In[34]:


#Distribution of current smokers with respect to age
plt.figure(figsize=(30,15), facecolor='w')
sns.countplot(x="age",data=data_frame,hue="currentSmoker")
plt.title("Graph showing which age group has more smokers.", size=30)
plt.xlabel("age", size=20)
plt.ylabel("age Count", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# ## Observation-
# Mid-age groups ranging from the age of 38 - 46 have more number of currentSmokers
# No currentSmokers observed below the age of 32
# maximum age for a currentSmokers is 70

# In[35]:


plt.figure(figsize=(10,10))
sns.violinplot(x='TenYearCHD', y='age', data=data_frame)


# In[36]:


#Plotting a linegraph to check the relationship between age and cigsPerDay, totChol, glucose.

var1 = data_frame.groupby("age").cigsPerDay.mean()
var2 = data_frame.groupby("age").totChol.mean()
var3 = data_frame.groupby("age").glucose.mean()

plt.figure(figsize=(16,10), facecolor='w')
sns.lineplot(data=var1, label="cigsPerDay")
sns.lineplot(data=var2, label="totChol")
sns.lineplot(data=var3, label="glucose")
plt.title("Graph showing totChol and cigsPerDay in every age group.", size=20)
plt.xlabel("age", size=20)
plt.ylabel("count", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# There is a minor relation between totChol and glucose. 
# cigsPerDay has a fairly parallel relationship with age.

# In[37]:


plt.figure(figsize=(8,8))
plt.pie(data_frame['TenYearCHD'].value_counts(), labels=['Neg.','Pos.'], autopct='%1.2f%%', explode=[0,0.2], shadow=True, colors=['crimson','gold'])
my_circle = plt.Circle( (0,0), 0.4, color='white')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Target Class Count')


# In[38]:


data_frame.head()


# In[39]:


# Dropping columns education
data_frame = data_frame.drop(['education'], axis=1)


# ## Version 1

# In[40]:


# Identify the features with the most importance for the outcome variable Heart Disease

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# separate independent & dependent variables
X = data_frame.iloc[:,0:14]  #independent columns
y = data_frame.iloc[:,-1]    

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(11,'Score'))  #print 10 best features


# In[41]:


featureScores = featureScores.sort_values(by='Score', ascending=False)
featureScores


# In[42]:


# visualizing feature selection
plt.figure(figsize=(20,5))
sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
plt.box(False)
plt.title('Feature importance', fontsize=16)
plt.xlabel('\n Features', fontsize=14)
plt.ylabel('Importance \n', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[43]:


# assign the all the column names, exept 'TenYearCHD', to columns 
columns = ['age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'prevalentStroke','BMI', 'glucose']

y = data_frame['TenYearCHD']
X = data_frame[columns]
# train and split the data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42, shuffle= False, stratify= None)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[44]:


print(len(X_train))
len(X_test)


# In[45]:


clf1 = LogisticRegression(random_state=42)

param_grid = { 
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

model_clf1 = GridSearchCV(estimator=clf1, param_grid=param_grid, cv= 10)
model_clf1.fit(X_train, y_train)


# In[46]:


model_clf1.best_params_


# In[47]:


clf1 = LogisticRegression(penalty = 'l1', solver = 'liblinear')
clf1.fit(X_train, y_train)
predictions1 = clf1.predict(X_test)
print(accuracy_score(y_test, predictions1))
print(cross_val_score(clf1, X_train, y_train, cv = 10, scoring = 'accuracy'))


# In[48]:


cm1 = confusion_matrix(y_test, predictions1) 
conf_matrix1 = pd.DataFrame(data = cm1,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[49]:


print(classification_report(y_test, predictions1))


# In[50]:


rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=7, 
                               n_jobs = -1)

rf_random.fit(X_train,y_train)


# In[51]:


rf_random.best_estimator_


# In[52]:


clf2 = RandomForestClassifier(criterion = 'gini', max_depth = 110, max_features = 'auto', n_estimators = 618, min_samples_split=5)
clf2.fit(X_train, y_train)
predictions2 = clf2.predict(X_test)
print(accuracy_score(y_test, predictions2))
print(cross_val_score(clf2, X_train, y_train, cv = 10, scoring = 'accuracy'))


# In[53]:


cm2 = confusion_matrix(y_test, predictions2) 
conf_matrix2 = pd.DataFrame(data = cm2,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix2, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[54]:


print(classification_report(y_test, predictions2))


# In[55]:


knn = KNeighborsClassifier(n_neighbors=15)
model3 = knn.fit(X_train, y_train)
prediction3 = knn.predict(X_test)
print(accuracy_score(y_test, prediction3))
print(cross_val_score(model3, X_train, y_train, cv = 10, scoring = 'accuracy'))


# In[56]:


cm3 = confusion_matrix(y_test, prediction3) 
conf_matrix3 = pd.DataFrame(data = cm3,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix3, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[57]:


print(classification_report(y_test, prediction3))


# In[58]:


#Number of trees
n_estimators = [int(i) for i in np.linspace(start=100,stop=1000,num=10)]
max_features = ['auto','sqrt']
max_depth = [int(i) for i in np.linspace(10, 100, num=10)]
max_depth.append(None)
min_samples_split=[2,5,10]
min_samples_leaf = [1,2,4]

#Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[59]:


gb=GradientBoostingClassifier(random_state=0)
#Random search of parameters, using 3 fold cross validation, 
#search across 100 different combinations
gb_random = RandomizedSearchCV(estimator=gb, param_distributions=random_grid,
                              n_iter=100, scoring='f1', 
                              cv=3, verbose=2, random_state=0, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
gb_random.fit(X_train,y_train)


# In[60]:


gb_random.best_estimator_


# In[61]:


model4 =  GradientBoostingClassifier(max_depth=20, max_features='auto', n_estimators=300,
                           random_state=0)
model4.fit(X_train,y_train)
prediction4 = model4.predict(X_test)
cm4 = confusion_matrix(y_test, prediction4)
gvc_acc_score = accuracy_score(y_test, prediction4)
print("confussion matrix")
print(cm4)


# In[62]:


cm4 = confusion_matrix(y_test, prediction4) 
conf_matrix4 = pd.DataFrame(data = cm4,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix4, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[63]:


print(classification_report(y_test,prediction4))


# In[64]:


classifier = SVC(random_state=42)

param_grid = { 
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5, 6],
    'gamma' : ['scale', 'auto']
}

gscv_clf5 = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 10)
gscv_clf5.fit(X_train, y_train)


# In[65]:


gscv_clf5.best_params_


# In[67]:


clf5 = SVC(degree = 2, gamma = 'scale', kernel= 'poly')
clf5.fit(X_train, y_train)
print(cross_val_score(clf5, X_train, y_train, cv = 10, scoring = 'accuracy'))
predictions5 = clf5.predict(X_test)


# In[68]:


cm5 = confusion_matrix(y_test, predictions5) 
conf_matrix5 = pd.DataFrame(data = cm5,  
            columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix5, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[69]:


print(classification_report(y_test, predictions5))


# In[70]:


from sklearn.tree import DecisionTreeClassifier
clf6 = DecisionTreeClassifier(min_samples_split=40, random_state=0) 
clf6.fit(X_train, y_train)
predictions6 = clf6.predict(X_test)


# In[71]:


cm6 = confusion_matrix(y_test, predictions6) 
conf_matrix6 = pd.DataFrame(data = cm6,  
            columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix6, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[72]:


print(classification_report(y_test, predictions6))


# In[73]:


lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,predictions1)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,prediction3)
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,predictions2)                                                             
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,predictions6)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,predictions5)
gbc_false_positive_rate,gbc_true_positive_rate,gbc_threshold = roc_curve(y_test,prediction4)

sns.set_style('whitegrid')
plt.figure(figsize=(15,8), facecolor='w')
plt.title('Reciever Operating Characterstic Curve')
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(gbc_false_positive_rate,gbc_true_positive_rate,label='Gradient Boosting Classifier')
plt.plot(svc_false_positive_rate,svc_true_positive_rate, label='Support Vector Classifier')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Decision Tree')

plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# ## Version 2

# In[74]:


target1=data_frame[data_frame['TenYearCHD']==1]
target0=data_frame[data_frame['TenYearCHD']==0]


# ## Resample Data

# In[75]:


from sklearn.utils import resample
target1=resample(target1,replace=True,n_samples=len(target0),random_state=40)
target=pd.concat([target0,target1])
target['TenYearCHD'].value_counts()


# In[76]:


df_data = target


# In[77]:


plt.figure(figsize=(12, 10), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sns.countplot(x="TenYearCHD", data=df_data)
plt.title("Count of TenYearCHD column", size=15)
plt.subplot(122)
labels=[0,1]
plt.pie(df_data["TenYearCHD"].value_counts(),autopct="%1.1f%%",labels=labels,colors=["crimson","yellow"])
plt.show()


# ##  Feature Selection

# In[78]:


# separate independent & dependent variables
X = df_data.iloc[:,0:14]  #independent columns
y = df_data.iloc[:,-1]    

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(11,'Score'))  #print 1 best features
print(X.shape)


# In[79]:


featureScores = featureScores.sort_values(by='Score', ascending=False)
featureScores


# In[80]:


plt.figure(figsize=(20,5))
sns.barplot(x='Specs', y='Score', data=featureScores, palette = "Blues_r")
plt.box(False)
plt.title('Feature importance', fontsize=15)
plt.xlabel('\n Features', fontsize=15)
plt.ylabel('Importance \n', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# ## Test - Train Split

# In[81]:


# assign the all the column names, exept 'TenYearCHD', to columns 
columns = ['age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'prevalentStroke','BMI', 'glucose']

y = df_data['TenYearCHD']
X = df_data[columns]
# train and split the data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
get_ipython().run_line_magic('store', 'X_train')
get_ipython().run_line_magic('store', 'X_test')


# ## Logistic Regression

# In[82]:


clf1_1= LogisticRegression(random_state=1)

param_grid = { 
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

model_clf1_1 = GridSearchCV(estimator=clf1_1, param_grid=param_grid)
model_clf1_1.fit(X_train, y_train)


# In[83]:


model_clf1_1.best_params_


# In[84]:


clf1_1 = LogisticRegression(penalty = 'l2', solver = 'newton-cg')
clf1_1.fit(X_train, y_train)
predictions1_1 = clf1_1.predict(X_test)
print(accuracy_score(y_test, predictions1_1))
print(cross_val_score(clf1_1, X_train, y_train,scoring = 'accuracy'))


# In[85]:


cm1_1 = confusion_matrix(y_test, predictions1_1) 
conf_matrix1_1 = pd.DataFrame(data = cm1_1,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1_1, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[86]:


print(classification_report(y_test, predictions1_1))


# ## Random Forest Classifier

# In[ ]:


rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=7, 
                               n_jobs = -1)

rf_random.fit(X_train,y_train)


# In[ ]:


rf_random.best_estimator_


# In[87]:


clf1_2 = RandomForestClassifier(criterion = 'gini', max_depth = 60, max_features = 'auto', n_estimators = 272)
clf1_2.fit(X_train, y_train)
predictions1_2 = clf1_2.predict(X_test)
print(accuracy_score(y_test, predictions1_2))
print(cross_val_score(clf1_2, X_train, y_train, cv = 3, scoring = 'accuracy'))


# In[88]:


cm1_2 = confusion_matrix(y_test, predictions1_2) 
conf_matrix1_2 = pd.DataFrame(data = cm1_2,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1_2, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[89]:


print(classification_report(y_test, predictions1_2))


# ## KNN Classifier

# In[90]:


param_grid = {'n_neighbors': [1,5,10,25,50,100],
              'weights': ['uniform','distance'],
              'algorithm': ['auto','ball_tree','kd_tree']}


# In[91]:


knn=KNeighborsClassifier()
#Random search of parameters, using 3 fold cross validation, 
#search across 100 different combinations
knn_random = GridSearchCV(knn,param_grid,cv=3,scoring='f1')

# Fit the random search model
knn_random.fit(X_train,y_train)


# In[92]:


knn_random.best_params_


# In[93]:


knn = KNeighborsClassifier(n_neighbors=1, weights='uniform',algorithm='auto')
model1_3 = knn.fit(X_train, y_train)
predictions1_3 = knn.predict(X_test)
print(accuracy_score(y_test, predictions1_3))
print(cross_val_score(model1_3, X_train, y_train, cv = 10, scoring = 'accuracy'))


# In[94]:


cm1_3 = confusion_matrix(y_test, predictions1_3) 
conf_matrix1_3 = pd.DataFrame(data = cm1_3,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1_3, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[95]:


print(classification_report(y_test, predictions1_3))


# ## Gradient Boosting Classifier

# In[ ]:


#Number of trees
n_estimators = [int(i) for i in np.linspace(start=100,stop=1000,num=10)]
max_features = ['auto','sqrt']
max_depth = [int(i) for i in np.linspace(10, 100, num=10)]
max_depth.append(None)
min_samples_split=[2,5,10]
min_samples_leaf = [1,2,4]

#Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[ ]:


gb=GradientBoostingClassifier(random_state=0)
#Random search of parameters, using 3 fold cross validation, 
#search across 100 different combinations
gb_random = RandomizedSearchCV(estimator=gb, param_distributions=random_grid,
                              n_iter=100, scoring='f1', 
                              cv=3, verbose=2, random_state=0, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
gb_random.fit(X_train,y_train)


# In[ ]:


gb_random.best_estimator_


# In[ ]:


import pickle


# In[96]:


model1_4 =  GradientBoostingClassifier(max_depth=40, max_features='sqrt', n_estimators=900,min_samples_split=5, random_state=0)
model1_4.fit(X_train,y_train)
prediction1_4 = model1_4.predict(X_test)
cm1_4 = confusion_matrix(y_test, prediction1_4)
gvc_acc_score = accuracy_score(y_test, prediction1_4)
filename = 'finalized_model.sav'
pickle.dump(model1_4, open(filename, 'wb'))


# In[97]:


cm1_4 = confusion_matrix(y_test, prediction1_4) 
conf_matrix1_4 = pd.DataFrame(data = cm1_4,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1_4, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[98]:


print(classification_report(y_test, prediction1_4))


# ##  Support Vector Classifier

# In[ ]:


classifier = SVC(random_state=42)

param_grid = { 
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5, 6],
    'gamma' : ['scale', 'auto']
}

gscv_clf1_5 = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 10)
gscv_clf1_5.fit(X_train, y_train)


# In[ ]:


gscv_clf1_5.best_params_


# In[99]:


clf1_5 = SVC(degree = 6, gamma = 'scale', kernel= 'poly')
clf1_5.fit(X_train, y_train)
print(cross_val_score(clf1_5, X_train, y_train, cv = 3, scoring = 'accuracy'))
prediction1_5 = clf1_5.predict(X_test)


# In[100]:


cm1_5 = confusion_matrix(y_test, prediction1_5) 
conf_matrix1_5 = pd.DataFrame(data = cm1_5,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1_5, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[101]:


print(classification_report(y_test, prediction1_5))


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
random_grid = {'criterion': ['gini', 'entropy'], 
               'max_depth': [1,5,10,20,50,100], 
               'max_features': [None],
               'min_samples_split': [2,5,10]}

dt=DecisionTreeClassifier(random_state=0)
#Random search of parameters, using 3 fold cross validation, 
#search across 100 different combinations
dt_random = RandomizedSearchCV(estimator=dt, param_distributions=random_grid,
                              n_iter=100, scoring='f1', 
                              cv=3, verbose=2, random_state=0, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
dt_random.fit(X_train,y_train)


# In[ ]:


dt_random.best_estimator_


# In[102]:


clf1_6 = DecisionTreeClassifier(criterion='entropy',max_depth=50, random_state=0) 
clf1_6.fit(X_train, y_train)
prediction1_6 = clf1_6.predict(X_test)


# In[103]:


cm1_6 = confusion_matrix(y_test, prediction1_6) 
conf_matrix1_6 = pd.DataFrame(data = cm1_6,  
            columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sns.heatmap(conf_matrix1_6, annot = True, fmt = 'd', cmap = "Blues") 
plt.show() 
  
print('The details for confusion matrix is =') 


# In[104]:


print(classification_report(y_test, prediction1_6))


# ## ROC Curves

# In[105]:


lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,predictions1_1)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,predictions1_3)
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,predictions1_2)                                                             
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,prediction1_6)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,prediction1_5)
gbc_false_positive_rate,gbc_true_positive_rate,gbc_threshold = roc_curve(y_test,prediction1_4)

sns.set_style('whitegrid')
plt.figure(figsize=(15,8), facecolor='w')
plt.title('Reciever Operating Characterstic Curve')
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(gbc_false_positive_rate,gbc_true_positive_rate,label='Gradient Boosting Classifier')
plt.plot(svc_false_positive_rate,svc_true_positive_rate, label='Support Vector Classifier')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Decision Tree')

plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# ## Feature Importance using SHAP
# 

# In[106]:


import shap 

#using shap for GBM
# Create Tree Explainer object that can calculate shap values
explainer = shap.TreeExplainer(model1_4)

columns = ['age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'prevalentStroke','BMI', 'glucose']

demo_shap_test = pd.DataFrame(data = X_test,
                             columns = columns)



# In[107]:


#for individual sample
choosen_instance = demo_shap_test.loc[[1]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], choosen_instance)


# In[108]:


shap_values = explainer.shap_values(demo_shap_test)


# In[109]:


#for explaning multiple
shap.force_plot(explainer.expected_value[0], shap_values[:200,:], demo_shap_test.iloc[:200,:])


# ## Feature Importance using LIME

# In[110]:


var_names = ['age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
            'diaBP', 'prevalentStroke','BMI', 'glucose']
var_= {'variable_name':var_names, "coefficient":model1_4.feature_importances_}
var_importance = pd.DataFrame(var_)


# In[111]:


model1_4.feature_importances_


# In[112]:


from lime import lime_tabular
from IPython.display import Image as IM
from IPython.display import clear_output
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import random


# In[113]:


explainer = LimeTabularExplainer(training_data= X_train, 
                                 mode= "classification", 
                                 feature_names= columns,
                                 verbose=True, 
                                 class_names=['0','1'])


# In[114]:


test_sample = X_test[148,:]
exp = explainer.explain_instance(data_row=test_sample,predict_fn= model1_4.predict_proba, num_features=12)
exp


# In[115]:


print('R²:' + str(exp.score))


# In[116]:


exp.local_exp


# In[117]:


plt = exp.as_pyplot_figure()
plt.tight_layout()


# In[118]:


coefs = pd.DataFrame(exp.as_list())[1].sum()
probability = coefs + exp.intercept[1]
print('The probability of class 1 is :', probability )
pd.DataFrame(exp.as_list())


# In[119]:


exp.show_in_notebook(show_table = True, show_all = False)


# In[120]:


print ('Explanation for class %s' % exp.class_names[0])
print ('\n'.join(map(str, exp.as_list())))


# In[121]:


d1 = {'age' : 'Age' , 'currentSmoker' : 'Smoking' ,
           'cigsPerDay' : 'High Smoking' , 'BPMeds' : 'BP Medication' ,
           'prevalentHyp' : 'Hypertension', 'diabetes' : 'Diabetes' ,
           'totChol' : 'Cholesterol' , 'sysBP' : 'Systolic Blood Pressure' ,
           'diaBP' : 'Diastolic Blood Pressure' , 'prevalentStroke' : 'Stroke' ,
            'BMI' : 'Body Mass Index' , 'glucose' : 'Glucose'}


# In[122]:


if (exp.predict_proba[1] > 0.50):
    high = 0
    col = ''
    for t in exp.as_list():
        if t[1] > high:
            high = t[1]
            ls = t[0].split()
            for i in ls:
                if i[0].isalpha():
                    col = i
                    break   
##    print(col)  
    marker = 1
    print("The algorithm predicts that you may have heart disease because of column :", d1.get(col))
    disease = d1.get(col)
else:
    print("The algorithm predicts that you may not have a heart disease")


# ## Disease Prevention Using Web Crawling

# In[123]:


disease_mapping = {'Age' : 'https://my.clevelandclinic.org/health/diseases/16891-heart-disease-adult-congenital-heart-disease' ,  
                   'BP Medication' : 'https://my.clevelandclinic.org/health/diseases/4314-hypertension-high-blood-pressure' , 
                   'Hypertension' : 'https://my.clevelandclinic.org/health/diseases/4314-hypertension-high-blood-pressure' , 
                   'Diabetes' : 'https://my.clevelandclinic.org/health/diseases/9812-diabetes-and-stroke' ,
                   'Cholesterol' : 'https://my.clevelandclinic.org/health/diseases/21656-hyperlipidemia' , 
                   'Systolic Blood Pressure' : 'https://my.clevelandclinic.org/health/diseases/4314-hypertension-high-blood-pressure' ,
                   'Diastolic Blood Pressure' : 'https://my.clevelandclinic.org/health/diseases/4314-hypertension-high-blood-pressure' , 
                    'Stroke' : 'https://my.clevelandclinic.org/health/diseases/5601-stroke' ,
                   'Glucose' : 'https://my.clevelandclinic.org/health/diseases/9815-hyperglycemia-high-blood-sugar'}


# In[124]:


disease_mapping_1 = {'High Smoking' : 
                     '''1. Get rid of all cigarettes and anything related to smoking, like lighters and ashtrays
                      2. When you get the urge to smoke, take a deep breath. Hold it for ten seconds and release it slowly. Repeat this several times until the urge to smoke is gone. You can also try meditation to reduce baseline stress levels.
                      3. Drink plenty of fluids, but limit caffeinated beverages and beverages containing alcohol
                      4. Don’t forget to exercise, because it has health benefits and help you relax.''' ,
                     'Smoking' :
                    '''1. Get rid of all cigarettes and anything related to smoking, like lighters and ashtrays
                      2. When you get the urge to smoke, take a deep breath. Hold it for ten seconds and release it slowly. Repeat this several times until the urge to smoke is gone. You can also try meditation to reduce baseline stress levels.
                      3. Drink plenty of fluids, but limit caffeinated beverages and beverages containing alcohol
                      4. Don’t forget to exercise, because it has health benefits and help you relax.''',
                     'Body Mass Index' : 
                     '''1 . 150 minutes a week of aerobic exercise can help reduce abdominal fat and overall obesity. That works out to 30 minutes of activity, five days a week. Choosing activities that you enjoy, such as brisk walking, dancing or swimming, can help you stay motivated.
                        2.  Eating fewer calories can help reduce abdominal fat. Changing your diet can also help you lose weight and improve overall obesity.
                        3. his diet includes eating mostly plant-based foods such as root and green vegetables, fresh fruits, legumes, nuts and whole grains, plus moderate servings of dairy, eggs, fish, lean poultry and seafood.'''}


# In[125]:


import bs4 as bs
import requests # For Web Scraping
from pprint import pprint 
if (marker == 1):
    if disease in disease_mapping:
        url = disease_mapping.get(disease)
        page = requests.get(url)
        soup = bs.BeautifulSoup(page.text)
        out = ""
        for i, val in enumerate(soup.find('div', class_ = 'js-section js-section--prevention').find_all('li')):
            pprint(f"{i+1}. {val.text}")
            out+=val.text
    elif disease in disease_mapping_1:
        output = disease_mapping_1.get(disease)
        print(output)


# In[ ]:





# In[ ]:




