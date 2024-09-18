#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/jbuja/ML_Streamline/survey_results_public.csv')


# In[2]:


df.head()


# In[3]:


df=df[df['ConvertedComp'].notnull()]


# In[4]:


df.columns
##df[df['CompFreq'] == 'Monthly']


# In[5]:


df.info()


# In[6]:


df.loc[df['CompFreq'] == 'Monthly', 'Salary'] = df['ConvertedComp'] 
df.loc[df['CompFreq'] == 'Yearly', 'Salary'] = df['ConvertedComp']


# In[7]:


df = df[["Country", "EdLevel", "YearsCodePro", "Employment",'CompFreq', "ConvertedComp",'Salary']]
#df = df.rename({"ConvertedComp": "Salary"}, axis=1)
df.head()


# In[8]:


df = df[df["Salary"].notnull()]
df.head()


# In[9]:


df.info()


# In[10]:


'''esto se podría hacer con sustituciones de missing values por promedios o strings más comunes,
pero porque hay renglones suficientes para correr el modelo no se hizo así'''

df = df.dropna()
df.isnull().sum()


# In[11]:


''' solo empleados tiempo completo'''

df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()


# In[12]:


df['Country'].value_counts()


# In[13]:


df['Country'].unique()


# In[14]:


'''recorta las observaciones a solo los países más importantes y crea la categoría otros''' 

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# In[15]:


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# In[16]:


country_map


# In[17]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[18]:


df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']


# In[19]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[20]:


df["YearsCodePro"].unique()


# In[21]:


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)


# In[22]:


df["EdLevel"].unique()


# In[23]:


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)


# In[24]:


df["EdLevel"].unique()


# In[25]:


from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()
#le.classes_


# In[26]:


le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df["Country"].unique()


# In[27]:


df=df.drop(["CompFreq","ConvertedComp"], axis=1)

df


# In[28]:


X = df.drop("Salary", axis=1)
y = df["Salary"]


# In[29]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)


# In[30]:


y_pred = linear_reg.predict(X)


# In[31]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))


# In[32]:


error


# In[33]:


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)


# In[34]:


y_pred = dec_tree_reg.predict(X)


# In[35]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[36]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)


# In[37]:


y_pred = random_forest_reg.predict(X)


# In[38]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[39]:


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)


# In[40]:


regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[41]:


X


# In[42]:


# country, edlevel, yearscode
X = np.array([["United States", 'Master’s degree', 15 ]])
X


# In[43]:


X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[44]:


y_pred = regressor.predict(X)
y_pred


# In[45]:


import pickle


# In[46]:


data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[47]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# In[48]:


y_pred = regressor_loaded.predict(X)
print(y_pred)


# In[49]:


print(X)


# In[ ]:




