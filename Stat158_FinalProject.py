#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf;


# In[3]:


data = pd.read_stata("EnosFowler_PivotalityExperiment.dta")


# In[4]:


data.head(1)


# ## Part 1 : Replicating the Original Analysis

# In[5]:


data["pivotal"].isna()


# In[6]:


data_no_missing_values = data.dropna(subset=["pivotal"])
data_no_missing_values[data_no_missing_values["stratum_id"] == 1]


# In[7]:


md = smf.ols("s11 ~ pivotal ", data_no_missing_values, groups=data_no_missing_values["stratum_id"])
mdf = md.fit(cov_type = "cluster", cov_kwds =  {'groups': data_no_missing_values['phone_id']})
print(mdf.summary())


# In[8]:


md_2 = smf.ols("s11 ~ pivotal ", data_no_missing_values[data_no_missing_values["ind_contact"]==1], groups=data_no_missing_values[data_no_missing_values["ind_contact"]==1]["stratum_id"])
mdf_2 = md_2.fit()
print(mdf_2.summary())



# ## Part 2: FRT under the Sharp Null with Difference-In-Means Test Statistic
# 

# In[9]:


dat = data_no_missing_values


# In[10]:


# number of n1 & n0 in the whole dataset
n1 = sum(dat["pivotal"] == 1)
n0 = sum(dat["pivotal"] == 0)
print(f"The overall number of treated units in this dataset is {n1} and the overall number of control units is {n0}")


# In[11]:


#calculating the prerequisite to run FRT under sharp null --> Is the proportion of control & treatment same across strata?
expectation = 0
for i in np.arange(44):
    expectation += (sum(dat[dat["stratum_id"] == (i+1)]["pivotal"] == 1) / n1) - (sum(dat[dat["stratum_id"] == (i+1)]["pivotal"] == 0) / n0)

round(expectation,3)


# Prerequisite Satisfied!

# In[12]:


#propensity score example
len(dat[dat["stratum_id"] == 1])/ len(dat)


# In[13]:


#per stratum treatment effect example
id_1 = dat[dat["stratum_id"] == 1]
np.mean(id_1[id_1["pivotal"] == 1]["s11"]) - np.mean(id_1[id_1["pivotal"] == 0]["s11"]) 


# In[14]:


#estimated_ate with difference-in-means as test statistic for intent to treat
estimated_ate = 0
for i in np.arange(44):
    propensity_score = len(dat[dat["stratum_id"] == (i+1)])/ len(dat)
    k = dat[dat["stratum_id"] == (i+1)]
    estimated_ate += propensity_score * (np.mean(k[k["pivotal"] == 1]["s11"]) - np.mean(k[k["pivotal"] == 0]["s11"]))
estimated_ate


# This estimated ATE is consistent with the regression coefficient estimate (the first one). 

# In[15]:


permuted_effects = []
for i in np.arange(200):
    permuted_ate = 0
    for i in np.arange(44):
        dat_stratum = dat[dat["stratum_id"] == (i + 1)]
        permuted_treatment = np.random.permutation(dat_stratum["pivotal"])
        dat_stratum["permuted_pivotal"] = permuted_treatment
        propensity_score = len(dat_stratum )/ len(dat)
        permuted_ate += propensity_score * (np.mean(dat_stratum[dat_stratum["permuted_pivotal"] == 1]["s11"]) - np.mean(dat_stratum[dat_stratum["permuted_pivotal"] == 0]["s11"]))
    permuted_effects.append(permuted_ate)

permuted_effects


# Calculate the p-value

# In[16]:


n = 0
for i in permuted_effects:
    if estimated_ate <= i:
        n+=1
n/200


# This p value is consistent with the authors' findings. 

# In[17]:


import matplotlib.pyplot as plt
plt.hist(permuted_effects, bins=10);
plt.axvline(x=estimated_ate, color='red')
plt.xlabel('Estimated ATE')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated ATE under the Null Hypothesis for Intent to Treat');


# In[18]:


#check prereq for contacted individual
contacted_ind = dat[dat["ind_contact"] == 1]

n1 = sum(contacted_ind["pivotal"] == 1)
n0 = sum(contacted_ind["pivotal"] == 0)

print(f"The overall number of treated units in this dataset is {n1} and the overall number of control units is {n0}")

expectation = 0
for i in np.arange(44):
    expectation += (sum(contacted_ind[contacted_ind["stratum_id"] == (i+1)]["pivotal"] == 1) / n1) - (sum(contacted_ind[contacted_ind["stratum_id"] == (i+1)]["pivotal"] == 0) / n0)

round(expectation,15)


# In[19]:


#estimated_ate with difference-in-means as test statistic for actually contacted individuals
estimated_ate_2 = 0
for i in np.arange(44):
    propensity_score = len(contacted_ind[contacted_ind["stratum_id"] == (i+1)])/ len(contacted_ind)
    k = contacted_ind[contacted_ind["stratum_id"] == (i+1)]
    estimated_ate_2 += propensity_score * (np.mean(k[k["pivotal"] == 1]["s11"]) - np.mean(k[k["pivotal"] == 0]["s11"]))
estimated_ate_2


# Consistent with the authors' results!

# In[20]:


permuted_effects_2 = []
for i in np.arange(100):
    permuted_ate_2 = 0
    for i in np.arange(44):
        dat_stratum = contacted_ind[contacted_ind["stratum_id"] == (i + 1)]
        permuted_treatment = np.random.permutation(dat_stratum["pivotal"])
        dat_stratum["permuted_pivotal"] = permuted_treatment
        propensity_score = len(dat_stratum )/ len(contacted_ind)
        permuted_ate_2 += propensity_score * (np.mean(dat_stratum[dat_stratum["permuted_pivotal"] == 1]["s11"]) - np.mean(dat_stratum[dat_stratum["permuted_pivotal"] == 0]["s11"]))
    permuted_effects_2.append(permuted_ate_2)

permuted_effects_2


# In[21]:


n = 0
for i in permuted_effects_2:
    if estimated_ate_2 <= i:
        n+=1
n/100


# In[22]:


plt.hist(permuted_effects_2, bins=20)
plt.axvline(x=estimated_ate_2, color='red')
plt.xlabel('Estimated ATE')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated ATE under the Null Hypothesis for Real Treatment for Contacted Individuals');


# This finding is consistent with the coefficient estimate of the authors. 

# ## Part 3
# ## Post Stratification : Matched Pair Experiment
# Inspired by George Box "Block what you can and randomize what you cannot"
# First analyze the data within strata. Maybe do exploratory data analysis for the data overall with hispanic population and also with the other variables that were not blocked
# Also whats up with the whole didn't actually answer my call thing.
# So, maybe actually evaluate the data with that.

# I will first match the data using the k nearest algorithm in python. While I do this I will make sure that the town names remain the same within each match considering that towns have an important role in voting outcomes.

# In[23]:


#mapping town names to floats so that 
np.unique(contacted_ind["town"].values)

contacted_ind["town_id"] = contacted_ind["town"].map({"CHARLTON": 100, "EAST BROOKFIELD": 200, "OXFORD": 300,
                                                     "SOUTHBRIDGE": 400, "SPENCER": 500})


# In[24]:


from sklearn.neighbors import NearestNeighbors


# In[25]:


df = contacted_ind.drop("town", axis = 1)
df = contacted_ind.drop("treatment", axis = 1)
df = df[["age", "g08", "town_id", "s09", "g10", "dem", "hispanic", "s11", "pivotal"]]
df


# In[26]:


treatment = df[df['pivotal'] == 1]
control = df[df['pivotal'] == 0]

control_filtered = control[control["town_id"] == treatment["town_id"].iloc[0]]


# Fit KNN model on control group
knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(control_filtered.drop(['pivotal', "town_id"], axis=1))

# Find nearest neighbors in control group for each observation in treatment group
distances, indices = knn.kneighbors(treatment.drop(['pivotal', "town_id"], axis=1))

# Create matched pairs dataframe
matches = pd.concat([
    treatment.reset_index(drop=True),
    control_filtered.iloc[indices.flatten()].reset_index(drop=True),
    pd.Series(distances.flatten(), name='distance')
], axis=1)
treated = matches["s11"].iloc[:, 0]
controlled = matches["s11"].iloc[:, 1]
matches.head()
#thefirsts11ispivotal_thesecondoneisnot


# In[36]:


ate_matched = 0
for i in np.arange(len(matches)):
    ate_matched += treated[i] - controlled[i]
ate_matched/len(matches)


# In[37]:


diff = treated - controlled
diff = np.array(diff)


# In[38]:


#paired_t_statistic
import scipy.stats as stats
from scipy.stats import ttest_rel

t, p = ttest_rel(np.array(controlled), np.array(treated))

(t, p)


# 

# In[41]:


# but we can use McNemar's statistic for binary outcome
from statsmodels.stats.contingency_tables import mcnemar


table =[[sum(treated), n- sum(treated)], [sum(controlled), n - sum(controlled)   ]]
result = mcnemar(table, exact = True, correction =True)


statistic = result.statistic
p_value = result.pvalue

# Print the results
print("McNemar's test statistic:", statistic)
print("p-value:", p_value)


# In[31]:


((n-sum(treated)) - sum(controlled))/(np.sqrt((n-sum(treated)) + sum(controlled)))


# In[32]:


#compare with normal
from scipy.stats import ttest_1samp
normal_dist = np.random.normal(0, 1, size=1000)
sdf = 0
for i in np.arange(1000):
    if 1.080634267190361 <= normal_dist[i]:
        sdf += 1
sdf/1000
#the p value is not statistically significant, again! Even in matched pair experiment, nevertheless we can see that 
#this estimator gives us a higher value for the treatment effects in the paper. --> 14%


# In[35]:


len(matches)


# In[ ]:





# ## Part 5: Regression Readjustment
# Let's see how are matched pair adjustment compares to regression adjustment

# In[33]:


contacted_ind


# In[46]:


model = smf.ols("s11 ~ pivotal + dem + g10 + age + hispanic + hispanic*pivotal + g10*pivotal + pivotal*age + s09 +s09*pivotal + g08*pivotal + g08", contacted_ind, groups=contacted_ind["stratum_id"])
model = model.fit()
print(model.summary())


# In[48]:


model = smf.ols("s11 ~ pivotal + dem + rep + age + hispanic + hispanic*pivotal + dem*pivotal + pivotal*age  ", contacted_ind, groups=contacted_ind["stratum_id"])
model = model.fit()
print(model.summary())


# In[42]:


matches["diff"] = treated - controlled


# In[44]:


model = sm.OLS(matches['diff'], sm.add_constant(matches[['dem', 'age', "hispanic", "s09", "g10", "town_id", "g08"]]))
# Get regression results
results = model.fit()

# Print summary of regression results
print(results.summary())


# In[ ]:




