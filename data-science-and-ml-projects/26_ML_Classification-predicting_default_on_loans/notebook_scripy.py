#!/usr/bin/env python
# coding: utf-8

# # Data Science Coding Test 

# The test is designed to be completed in 4 hours, which is also the deadline. There are three problems  in total. The main evaluation criteria are accuracy and completeness: please try to do as many tasks as possible within the allocated time. 
# 
# You can use either __R__ or __Python__ and you MUST deliver all the codes, accompanied by estimation results and graphs/tables required in this test in the format of your choice. You can leverage any packages that you find useful for the accomplishment of the test.

# ## Problem 1

# Use file __“mortgages.csv”__ and __“macros.csv”__ for this exercise. The brief description of the variables are given in the table below. The two datasets have a panel and time-series structure, respectively.

# __mortgages.csv__
# 
# Column | Type | Description
# :---|:---|:---
# `loan_id` | Numeric | The unique ID assigned to every loan application.
# `time` | Numeric| Observation month in elapsed time format (# of months since 1960-Jan)
# `obs_month` | Numeric | Observation month in YYYYMM format
# `orig_date` | Date | Loan origination date
# `orig_date_elapsed` | Numeric | Loan origination date in elapsed time format (# of months since 1960-Jan)
# `maturity_date` | Date | Loan maturity date
# `occupancy` | Categorical | Occupancy type of the collateral
# `mortgage_type` | Categorical | Mortgage type
# `arrears` | Numeric | Number of months a loan is in arrears
# `current_balance` | Numeric | Current outstanding balance of the loan
# `current_ltv` | Numeric | Current loan-to-value ratio
# `credit_score` | Numeric | Origination credit score of the borrower
# `original_ltv` | Numeric | Origination loan-to-value ratio (LTV)
# `hpi_o` | Numeric | House price index at loan origination date
# `interest_rate_o` | Numeric | Mortgage rate at loan origination date
# 
# __macros.csv__
# 
# Column | Type | Description
# :---|:---|:---
# `hpi_t` | Numeric | House price index at current date
# `interest_rate_t` | Numeric | Mortgage rate at current date
# `gdp_t` | Numeric | GDP annual growth rate at current date

# 1. __How many observations and how many loans (*loan_id*) are there in the dataset? What is the time coverage (start and end dates) in terms of observation month, origination date and the maturity date?__

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mortgages.csv')


# In[57]:


# Converting date columns to date format
for x in ['obs_month', 'orig_date', 'maturity_date']:
    df[x] = pd.to_datetime(df[x], format='%Y%m')

df.head()


# In[58]:


print('Number of unique loans:')
print(df['loan_id'].unique().shape[0]) # Unique loan_id(s)
print()
print('Number of observations:')
print(df.shape[0]) # Number of observations
print()

# Print the min value of the obs_month, orig_date, maturity date
print('Start date of obs_month:')
print(df['obs_month'].min()) # Min value of obs_month
print()

print('Start date of orig_date:')
print(df['orig_date'].min()) # Min value of orig_date
print()

print('Start date of maturity_date:')
print(df['maturity_date'].min()) # Min value of maturity_date
print()

print('End date of obs_month:')
print(df['obs_month'].max()) # Max value of obs_month
print()

print('End date of orig_date:')
print(df['orig_date'].max()) # Max value of orig_date
print()

print('End date of maturity_date:')
print(df['maturity_date'].max()) # Max value of maturity_date


# 2. __Create a table reporting the number of loans and percentage of loans per month. Create a distribution plot (bar plot) showing the number of loans at each observation date in the dataset.__

# In[59]:


# Option 1 (if we want to have both year and month in the table)

# Extract month and year
df['month'] = df['obs_month'].dt.month
df['year'] = df['obs_month'].dt.year

# Group by month and calculate the number of loans per month
year_monthly_loans = df.groupby(['year', 'month']).size().reset_index(name='Number of Loans')

# Calculate the total loans per year for percentage calculation
total_loans_per_year = df.groupby('year').size()

# Calculate the percentage of loans per month and year
year_monthly_loans['Percentage of Loans (%)'] = (year_monthly_loans['Number of Loans'] / year_monthly_loans.apply(lambda row: total_loans_per_year[row['year']], axis=1)) * 100

# Display the table
year_monthly_loans.head()


# In[60]:


# Define the month names for the x-axis labels
month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Create a pivot table for the monthly loans data
pivot_table = year_monthly_loans.pivot(index='year', columns='month', values='Number of Loans')

# Create a bar plot
plt.figure(figsize=(10, 6))
pivot_table.plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Number of Loans')
plt.title('Figure 1.1. Distribution of Loans by Month and Year')
plt.xticks(rotation=0)
plt.legend(title='Month', labels=month_names)
plt.tight_layout()

# Show the plot
plt.show()


# In[61]:


# Option 2. If we want to have only month in the table

# Group by month and calculate the number of loans per month
monthly_loans = df.groupby('month').size().reset_index(name='Number of Loans')

# Calculate the total loans for proportion calculation
total_loans = len(df)

# Calculate the proportion of loans per month
monthly_loans['Proportion of Loans (%)'] = (monthly_loans['Number of Loans'] / total_loans) * 100

# Display the table
print(monthly_loans)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(monthly_loans['month'], monthly_loans['Number of Loans'])
plt.xlabel('Month')
plt.ylabel('Number of Loans')
plt.title('Figure 1.2. Number of Loans by each Month')
plt.xticks(monthly_loans['month'], month_names)  # Replace numeric ticks with month names
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# 3. __Create a table reporting the number and share of observations per vintage (origination year) using the variable _orig_date_. Create a distribution plot (bar plot) showing the number of loans at each vintage in the dataset__

# In[62]:


# Extract origination year
df['origination_year'] = df['orig_date'].dt.year

# Group by origination year and calculate the number of loans per vintage
vintage_loans = df.groupby('origination_year').size().reset_index(name='Number of Loans')

# Calculate the total loans for percentage calculation
total_loans = len(df)

# Calculate the percentage of loans per vintage
vintage_loans['Share of Loans (%)'] = (vintage_loans['Number of Loans'] / total_loans) * 100

# Create a table
print(vintage_loans)


# In[63]:


# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(vintage_loans['origination_year'], vintage_loans['Number of Loans'])
plt.xlabel('Origination Year')
plt.ylabel('Number of Loans')
plt.title('Figure 2. Distribution of Loans by Vintage')
plt.xticks(vintage_loans['origination_year'], rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()


# In[64]:


# It looks like there aren't any loans originated in 2011, 2017, and 2019


# 4. __Create a new variable called “*first_obs*” that is the first observation date of each account. Create and summarize the observation lag (difference between the origination date and the first observation date) in the dataset.__

# In[65]:


# Group by 'loan_id' and calculate the first observation date for each account
df['first_obs'] = df.groupby('loan_id')['obs_month'].transform('min')

# Calculate the observation lag (difference between 'orig_date' and 'first_obs')
df['obs_lag'] = (df['orig_date'] - df['first_obs']).dt.days

# Convert df['obs_lag'] to asbolute values
df['obs_lag'] = df['obs_lag'].abs()

df.head()


# 5. __Merge the macro variables (*interest_rate_t, hpi_t, gdp_t*) provided in the macro.csv file with the mortgages.csv file, using the variable *obs_month*.__

# In[66]:


macro = pd.read_csv('macros.csv')

# Ensuring correct format
macro['date'] = pd.to_datetime(macro['date'], format='%Y%m')

macro.head()


# In[67]:


# Using inner join
df_merged = pd.merge(left=df, right=macro, left_on='obs_month', right_on='date')
df_merged.shape


# In[68]:


df_merged.head()


# 6. __In credit risk vocabulary, a loan that is late on repayments is said to be in arrears, measured in months of non-payment. Using the dataset created in step 5, create an indicator variable (*default_flag*) taking the value of 1 when a loan becomes 3 or more months in arrears, and zero otherwise. The variable “arrears” indicates the number of months in arrears.__

# In[69]:


# Adding the new column to a specified location using insert(loc, col, condition)
df_merged.insert(9, 'default_flag', np.where(df_merged['arrears'] >= 3, 1, 0))
df_merged.head()


# In[70]:


df_merged['arrears'].value_counts()


# In[71]:


# Sanity check
df_merged['default_flag'].value_counts()


# 7. __Create a single table reporting the summary statistics (mean, median, std, min and max) of the *default flag, current balance, current LTV, borrower credit score, original LTV, house price index at origination and the current house price index*, where each row of the table reports the statistics for one variable. Add a column to the table, reporting the number of missing observations for each variable__

# In[72]:


df_merged.columns


# In[73]:


missing_count = df_merged.isnull().sum()
missing_count


# In[74]:


def describe(df, stats):
    d = df.describe()
    return d.append(df.agg(stats))

df_describe = describe(df_merged, ['median'])

# Add a new row to the describe statistics for missing values count
missing_row = pd.DataFrame(missing_count, columns=['missing_count']).T
result_with_missing = pd.concat([df_describe, missing_row], sort=False)  # Add sort parameter here to prevent warining

# drop the columns that are not needed
result_with_missing = result_with_missing[[f for f in list(df_describe) if f in ['default_flag', 'current_balance', 'current_ltv', 'credit_score', 'original_ltv', 'hpi_o', 'hpi_t']]]

# Show only first 4 rows of describe table
result_with_missing = result_with_missing.T

result_with_missing.drop(['count', '25%', '50%', '75%'], axis=1, inplace=True)

result_with_missing.fillna(0, inplace=True)

result_with_missing


# 8. __Tabulate the variables: *arrears, mortgage_type and occupancy* to show the number and the percentage of observations in each category of these variables__

# In[75]:


# arrears table
arrears_tab = df['arrears'].value_counts()
arrears_percentage = (arrears_tab / arrears_tab.sum()) * 100
arrears_summary = pd.DataFrame({
    'Arrears': arrears_tab,
    'Arrears (%)': arrears_percentage
})
arrears_summary


# In[76]:


# occupancy table
occupancy_tab = df['occupancy'].value_counts()
occupancy_percentage = (occupancy_tab / occupancy_tab.sum()) * 100
occupancy_summary = pd.DataFrame({
    'Occupancy': occupancy_tab,
    'Occupancy (%)': occupancy_percentage
})
occupancy_summary


# In[77]:


# mortgage_type table
mortgage_type_tab = df['mortgage_type'].value_counts()
mortgage_type_percentage = (mortgage_type_tab / mortgage_type_tab.sum()) * 100
mortgage_type_summary = pd.DataFrame({
    'Mortgage Type': mortgage_type_tab,
    'Mortgage Type (%)': mortgage_type_percentage
})
mortgage_type_summary


# 9. __Based on the findings in points 7 and 8, make the corresponding treatments of missing values (e.g. imputation, drop).__

# In[78]:


missing_count


# In[79]:


# Four columns to treat, since there is not much time I will impute the mean for numeric and drop non-numeric

#  imputation: 'credit_score', 'current_ltv', 
df_merged['credit_score'] = df_merged['credit_score'].fillna(df_merged['credit_score'].mean())
df_merged['current_ltv'] = df_merged['current_ltv'].fillna(df_merged['current_ltv'].mean())


# In[80]:


#  dropping: 'mortgage_type', 'occupancy' 
df_filtered = df_merged.dropna()
print(df_merged.shape)
print(df_filtered.shape)


# In[81]:


# We lost some data.


# 10. __Next, partition the dataset into development and test samples, this should be a random 80/20 split (80% development / 20% test). You can use the seed (1219) for generating this split.__
# 
# 11. __After you have your development sample you should estimate a logistic model using *default_flag* as your dependent and *credit_score, occupancy,  current_ltv, mortgage_type, gdp_t* and “change in hpi since origination” (name it *d_hpi*) as your independent variables.__
# 
# 
# __ __
# 
# __My Notes:__
# - There is a bit of a overlap between exercise 10 and 11 because for a logistic model we need to encode some variables, and we need to add a new variable, so I merged instructions for 10 and 11 in this markdown cell.

# In[82]:


# Setting seed which is used in the cells below
seed = 1219

# Creating variable 'd_hpi'
df_filtered.insert(24, 'd_hpi', (df_filtered['hpi_o'] - df_filtered['hpi_t']))
df_filtered


# In[83]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Assuming you have a DataFrame 'df_filtered' with the required columns
features = ['credit_score', 'current_ltv', 'gdp_t', 'd_hpi', 'mortgage_type', 'occupancy', 'obs_month'] #independent variables

dependent_variable = 'default_flag'

df_filtered = df_filtered[features + [dependent_variable]]

# ENCODING
df_encoded = pd.get_dummies(df_filtered[['mortgage_type', 'occupancy']], drop_first=True)
df_model = pd.concat([df_filtered, df_encoded], axis=1)
df_model.drop(columns=['mortgage_type', 'occupancy'], inplace=True)

df_model.head()

# df_model.isna().sum() # sanity check


# In[84]:


features_encoded = ['credit_score',
                    'current_ltv',
                    'gdp_t',
                    'd_hpi',
                    'mortgage_type_erm',
                    'mortgage_type_linear',
                    'occupancy_holiday home',
                    'occupancy_owner occupied'
                   ]

# Split the data into development (train) and test sets
X = df_model[features_encoded]
y = df_model[dependent_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Create and train the logistic regression model
logreg_model = LogisticRegression(solver = 'lbfgs') # Added solver to prevent the warning
logreg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# 12. __Interpret the coefficient on the current LTV variable in this model.__

# In[85]:


# Extract the index of 'current_ltv' in the features list
ltv_index = features_encoded.index('current_ltv') 

# Get the coefficient for 'current_ltv'
ltv_coefficient = logreg_model.coef_[0][ltv_index]

# Calculate the odds ratio
odds_ratio = np.exp(ltv_coefficient)

print("Coefficient for current LTV:", ltv_coefficient)
print("Odds Ratio:", odds_ratio)


# In[86]:


# Interpretation
print(f'Since odds ratio > 1, for a one-unit increase in current LTV, the odds of default increase by approximately {round((odds_ratio - 1) * 100, 2)}%.')


# 13. __Generate predicted default rates from this model. Name the variable *“pred_default”*.__
# 
# __My notes:__
# - Since in exercise 15 I am asked to generate predictions on test, I am assuming here we want to generate predictions on train dataset and compare

# In[87]:


# Predict probabilities for the test set
pred_default = logreg_model.predict_proba(X_train)[:, 1]

pred_default


# In[88]:


# Create a new DataFrame with actual default flag and predicted default probabilities
pred_default_df = pd.DataFrame({'default_flag': y_train, 'pred_default': pred_default})

# Display the predicted default rates
print(pred_default_df)


# 14. __In-sample model validation:__
# 
#     - Plot average actual predicted default rates alongside the average predicted default rates over time (by observation month) in a line graph. Interpret the model performance based on the result
#     - Create buckets of current LTV going from 1 to maximum current LTV, by increments of 1 (1,2,3,…). Plot (line graph) average actual default rates alongside the average predicted default rates by LTV buckets (x-axis). How does it look? Does the trend you observe obey the implications of the coefficient of LTV? Does the model do a good job in accounting for the effect of ltv
#     - Create 3 categories of borrower credit score as follows:
#         - low-score: credit_score <= 650
#         - mid-score: 650 < credit_score <= 800
#         - high-score: credit_score > 800
#     - Create a single bar plot, comparing the average actual default rate to average predicted default rate, (side by side) per credit score category

# __Part 1__

# In[89]:


# Merge the 'obs_month' column from the original df_model into pred_default_df
pred_default_df = pred_default_df.merge(df_model[['obs_month']], left_index=True, right_index=True)

pred_default_df


# In[90]:


# Calculate average actual and predicted default rates over time (by observation month)
average_actual_default_by_month = pred_default_df.groupby('obs_month')['default_flag'].mean()
average_predicted_default_by_month = pred_default_df.groupby('obs_month')['pred_default'].mean()

# Plot
plt.plot(average_actual_default_by_month.index, average_actual_default_by_month, label='Average Actual Default Rate')
plt.plot(average_predicted_default_by_month.index, average_predicted_default_by_month, label='Average Predicted Default Rate')
plt.xlabel('Observation Month')
plt.ylabel('Default Rate')
plt.title('Figure 3. Average Actual vs. Predicted Default Rates Over Time')
plt.legend()
plt.show()


# __Part 1 interpretation:__ 
# - With the exception of a few spikes the predicted Average Default Rate curve closely fits the Actual Average Default Rate curve. 
# - However, between 2014 and 2015 1/2, possibly there are a few outliers because of the spike in the Actual Default Rate

# __Part 2__
# 
# - Create buckets of current LTV going from 1 to maximum current LTV, by increments of 1 (1,2,3,…). Plot (line graph) average actual default rates alongside the average predicted default rates by LTV buckets (x-axis). How does it look? Does the trend you observe obey the implications of the coefficient of LTV? Does the model do a good job in accounting for the effect of ltv

# In[91]:


# Create buckets of current LTV
ltv_buckets = np.arange(1, int(df_model['current_ltv'].max()) + 1, 1)

ltv_buckets

# Add a new column to the DataFrame with LTV buckets
df_model['ltv_bucket'] = pd.cut(df_model['current_ltv'], bins=ltv_buckets)

# Add the 'pred_default' column from pred_default_df to df_model
df_model['pred_default'] = pred_default_df['pred_default']

# Calculate average actual and predicted default rates by LTV buckets
average_actual_default_by_ltv = df_model.groupby('ltv_bucket')['default_flag'].mean()
average_predicted_default_by_ltv = df_model.groupby('ltv_bucket')['pred_default'].mean()

# Reindex the DataFrames to ensure alignment
average_actual_default_by_ltv = average_actual_default_by_ltv.reindex(ltv_buckets)
average_predicted_default_by_ltv = average_predicted_default_by_ltv.reindex(ltv_buckets)

# Plot
plt.plot(ltv_buckets, average_actual_default_by_ltv, label='Average Actual Default Rate')
plt.plot(ltv_buckets, average_predicted_default_by_ltv, label='Average Predicted Default Rate')
plt.xlabel('Current LTV Buckets')
plt.ylabel('Default Rate')
plt.title('Figure 6. Average Actual vs. Predicted Default Rates by Current LTV Buckets')
plt.legend()

# Set x-axis limits to show only values from 0 to 200
# plt.xlim(0, 170)

plt.show()


# __Part 2 interpretation:__
# - The curve looks good but we should probably take care of the outliers for Current LTV
# - Figure 7 below shows the distribution of outliers for current_ltv

# In[92]:


df_model['current_ltv'].describe()


# In[93]:


# Plot current_ltv using boxplot to see outliers
sns.boxplot(x=df_model['current_ltv'])
plt.title('Figure 7. Distribution of current_ltv')


# __Part 3__
# 
# - Create 3 categories of borrower credit score as follows:
#     - low-score: credit_score <= 650
#     - mid-score: 650 < credit_score <= 800
#     - high-score: credit_score > 800
#  
# - Create a single bar plot, comparing the average actual default rate to average predicted default rate, (side by side) per credit score category

# In[94]:


# Create categories based on credit score
def categorize_credit_score(score):
    if score <= 650:
        return 'low-score'
    elif score <= 800:
        return 'mid-score'
    else:
        return 'high-score'

df_model.head()


# In[95]:


df_model['credit_score_category'] = df_model['credit_score'].apply(categorize_credit_score)

# Calculate average actual and predicted default rates by credit score categories
average_actual_default_by_score = df_model.groupby('credit_score_category')['default_flag'].mean()
average_predicted_default_by_score = df_model.groupby('credit_score_category')['pred_default'].mean()

# Plot
average_actual_default_by_score.plot(kind='bar', color='blue', alpha=0.5, label='Average Actual Default Rate')
average_predicted_default_by_score.plot(kind='bar', color='orange', alpha=0.5, label='Average Predicted Default Rate')
plt.xlabel('Credit Score Category')
plt.ylabel('Default Rate')
plt.title('Figure 8. Average Actual vs. Predicted Default Rates by Credit Score Category')
plt.legend()
plt.show()


# 15. __Generate predictions using the test sample you created, and a plot, as in question 14.a, for the test sample actual and predicted default rates over time. How does the model behave on test sample.__

# In[96]:


# Generate predicted default probabilities using the logistic regression model
test_pred_default_probabilities = logreg_model.predict_proba(X_test)[:, 1]


# In[97]:


# Create a new DataFrame with actual default flag and predicted default probabilities
test_pred_default_probabilities_df = pd.DataFrame({'default_flag': y_test, 'pred_default': test_pred_default_probabilities})

# Display the predicted default rates
# print(test_pred_default_probabilities_df)

# Merge the 'obs_month' column from the original df_model into pred_default_df
test_pred_default_probabilities_df = test_pred_default_probabilities_df.merge(df_model[['obs_month']], left_index=True, right_index=True)

test_pred_default_probabilities_df


# In[98]:


# Plotting as in 14a

# Calculate average actual default rates and average predicted default rates over time
average_actual_default_over_time = test_pred_default_probabilities_df.groupby('obs_month')['default_flag'].mean()
average_predicted_default_over_time = test_pred_default_probabilities_df.groupby('obs_month')['pred_default'].mean()

# Plot
plt.plot(average_actual_default_over_time.index, average_actual_default_over_time, label='Average Actual Default Rate')
plt.plot(average_predicted_default_over_time.index, average_predicted_default_over_time, label='Average Predicted Default Rate')
plt.xlabel('Observation Month')
plt.ylabel('Default Rate')
plt.title('Figure 9. Average Actual vs. Predicted Default Rates Over Time (Test Sample)')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()


# 16. __Assuming any predicted value of the dependent variable above 0.1 to indicate default, create an indicator variable (outcome) showing whether or not a loan is in default, according to the model. Use the test sample (the one containing 20% of the observations)__

# In[99]:


# Set the threshold for predicting default
default_threshold = 0.1

# Create an indicator variable for predicted default based on the threshold
test_pred_default_probabilities_df['outcome'] = (test_pred_default_probabilities_df['pred_default'] > default_threshold).astype(int)

# Display the updated DataFrame with the predicted default indicator variable
test_pred_default_probabilities_df[['default_flag', 'pred_default', 'outcome']]


# In[100]:


test_pred_default_probabilities_df['outcome'].value_counts()


# 17. __Create a confusion matrix to report the results of comparing the predicted observations of default with actual observations of default in the test sample.__

# In[101]:


from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_pred_default_probabilities_df['default_flag'], 
                               test_pred_default_probabilities_df['outcome'])

# Extract confusion matrix values
true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()

# Print the confusion matrix values
print("Confusion Matrix:")
print("True Negative:", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print("True Positive:", true_positive)


# In[102]:


# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('figure 10. Confusion Matrix')
plt.show()


# 18. __Fit a Decision Tree on the development sample using Grid Search for the following parameters (use random state 1219):__ 
#     - criterion: "gini" or "entropy"
#     - max_depth: from 1 to 5 with step 1
#     - min_samples_leaf: from 0.1 to 0.5 with step 0.2
#     - max_features: from 0.1 to 0.5 with step 0.1

# In[103]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Set random seed
seed = 1219

# Define parameter grid
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 6),
    'min_samples_leaf': [0.1, 0.3, 0.5],
    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Create Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=seed)

# Create GridSearchCV instance
grid_search_dt = GridSearchCV(dt_classifier, param_grid_dt, cv=5)

# Fit the model on development sample
grid_search_dt.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search_dt.best_params_)


# 19. __Fit a XGBoost model on the development sample using Grid Search for the following parameters (use random state 1219):__
# 
#     - loss: "deviance"
#     - n_estimators: from 50 to 150 with step 50
#     - learning_rate: from 0.5 to 1 with step 0.1
#     - max_depth: from 1 to 5 with step 1
#     - max_features: from 0.1 to 0.5 with step 0.1

# ## Important note:
# 
# __As you can see below, I executed the grid_search with the parameters specified in the task, but the kernel was running with a lot of warnings and after about 12 mins (see the warning logs) I decided I can't waste more time.__
# 
# __I think loss and max_features parameters are no longer used in xgboost__
# 
# __I will run the grid search again but for less parameters__ 

# In[50]:


# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# # Create a parameter grid for Grid Search
# param_grid_xgb = {
#     'n_estimators': [50, 100, 150],
#     'learning_rate': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'max_depth': [1, 2, 3, 4, 5],
#     'max_features': [0.1, 0.2, 0.3, 0.4, 0.5]
# }

# # Initialize the XGBoost classifier
# xgb_model = XGBClassifier(loss='deviance', random_state=1219)

# # Initialize Grid Search with cross-validation
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='accuracy', cv=5, verbose=1)

# # Fit the Grid Search to the development sample
# grid_search.fit(X_train, y_train)

# # Print the best parameters found by Grid Search
# print("Best Parameters:", grid_search.best_params_)


# 
# ## Since the grid search failed I decided I will skip this and fit a XGBooost model with default params

# In[104]:


from xgboost import XGBClassifier

# Create and fit the XGBoost classifier model
xgb_model = XGBClassifier(random_state=1219)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_xgb))


# 20. __Select the best models from Grid Search from Step 18 and 19, make predictions on test sample using those models and construct the Confusion Matrix for both models. Based on the results, which model would you choose to use in production? Please provide detailsof reasoning for model selection.__

# In[105]:


from sklearn.metrics import accuracy_score

# Get the best parameters from Decision Tree grid search
best_dt_params = grid_search_dt.best_params_

# Create the best Decision Tree classifier
best_dt_classifier = DecisionTreeClassifier(**best_dt_params, random_state=seed)

# Fit the best Decision Tree classifier on the development sample
best_dt_classifier.fit(X_train, y_train)

# Predict using the best Decision Tree and kNN classifiers on the validation sample
dt_val_predictions = best_dt_classifier.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_val_predictions)
# dt_accuracy

# Create a confusion matrix for DT
conf_matrix_dt = confusion_matrix(y_test, dt_val_predictions)

print("Confusion Matrix for DecisionTree Model:")
print(conf_matrix_dt)


# In[107]:


dt_accuracy


# In[106]:


# Create a confusion matrix for XGBoost
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

print("Confusion Matrix for XGBoost Model:")
print(conf_matrix_xgb)


# In[108]:


xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_accuracy


# __Answer:__
# 
# - Since the result for both models is exactly the same, I assume I commited an error somewhere in the previous tasks
# - The logistic regression model predicts probabilities and by applying an arbitrary threshold we can tweak the final binary result
# - For that reason, from the three fitted models I personally would select the Logistic Regression model
# - However, if we were to assume that DT and XGBoost are predicting correctly, the model works fine only with the negative outcomes, because second column of the confusion matrix has 0 for both False Positives and True Positives. 
# - If I were to select from these two models (DT and XGBoost), I would select XGboost because it combines several DTs and if more data were introduced for traininig it would definitely predict better and would not cause significant overfitting

# ## Problem 2

# Use file *“macros2.xlsx”* this exercise. The brief description of the variables are given in the table below. The dataset has a time-series structure

# __macros2.xlsx__
# 
# Column | Type | Description
# :---|:---|:---
# `Period` | Date | Observation date in YYYYQQ format (Year Quarter)
# `HPI_S0` | Numeric | HPI under S0 Scenario forecast
# `HPI_S1` | Numeric | HPI under S1 Scenario forecast
# `HPI_S3` | Numeric | HPI under S3 Scenario forecast
# `HPI_S4` | Numeric | HPI under S4 Scenario forecast
# `LBR_S0` | Numeric | Unemployment Rate under S0 Scenario
# `LBR_S1` | Numeric | Unemployment Rate under S1 Scenario
# `LBR_S3` | Numeric | Unemployment Rate under S3 Scenario
# `LBR_S4` | Numeric | Unemployment Rate under S4 Scenario

# 1. __Transform the data in a way to have 4 columns corresponding to date variable, values for HPI and unemployment rate (in one column), variable category (either HPI or LBR) and scenario name (e.g. S0, S1, etc.), as shown below.__

# Date | Value | Category | Scenario
# :---|:---:|:---:|:---:
# 2010Q1 | 150 | HPI | S0
# 2010Q1 | 152 | HPI | S1
# 2010Q1 | 145 | HPI | S3
# 2010Q1 | 130 | HPI | S4
# 2010Q1 | 2.15 | LBR | S0

# In[143]:


data = pd.ExcelFile('macros2.xlsx')
data.sheet_names


# In[144]:


df_xlsx = pd.read_excel('macros2.xlsx', sheet_name = 'Sheet1')
df_xlsx.columns


# In[145]:


# df_xlsx


# In[146]:


# Create a list of relevant columns
columns_to_melt = ['Period', 'HPI_S0', 'HPI_S1', 'HPI_S3', 'HPI_S4', 'LBR_S0', 'LBR_S1', 'LBR_S3', 'LBR_S4']

# Initialize an empty list to store the melted data
melted_data = []

# Iterate through each row in the dataframe
for index, row in df_xlsx.iterrows():
    period = row['Period']
    for col in columns_to_melt[1:]:  # Skip 'Period' column
        category, scenario = col.split('_')
        value = row[col]
        melted_data.append([period, value, category, scenario])

# Create a new dataframe from the melted data
df_melted = pd.DataFrame(melted_data, columns=['Period', 'Value', 'Category', 'Scenario'])
        
# Drop the index column
df_melted = df_melted.rename(columns={"Period": "Date"})

df_melted


# 2. __Based on data obtained in question 1, transform the data in a way to obtain 4 columns, but now with the following variables, date variable, HPI values, Unemployment values and scenario name (e.g. S0, S1, etc.), as shown below.__

# Date | HPI | Unemployment | Scenario
# :---|:---:|:---:|:---:
# 2010Q1 | 150 | 2.15 | S0
# 2010Q1 | 155 | 2.10 | S1
# 2010Q1 | 145 | 2.5 | S3
# 2010Q1 | 130 | 3 | S4
# 2010Q2 | 152 | 2.14 | S0

# In[147]:


# Create a list of relevant columns
columns_to_melt = ['Period', 'HPI_S0', 'HPI_S1', 'HPI_S3', 'HPI_S4', 'LBR_S0', 'LBR_S1', 'LBR_S3', 'LBR_S4']

# Initialize an empty list to store the transformed data
transformed_data = []

# Iterate through each row in the dataframe
for index, row in df_xlsx.iterrows():
    period = row['Period']
    for col in columns_to_melt[1:]:  # Skip 'Period' column
        scenario = col.split('_')[1]  # Extract scenario
        hpi_value = row[col]  # Extract HPI value
        unemployment_column = f"LBR_{scenario}"  # Corresponding Unemployment column
        unemployment_value = row[unemployment_column]  # Extract Unemployment value
        transformed_data.append([period, hpi_value, unemployment_value, scenario])

# Create a new dataframe from the transformed data
df_transformed = pd.DataFrame(transformed_data, columns=['Date', 'HPI', 'Unemployment', 'Scenario'])

# Print the final dataframe
df_transformed


# 3. __Create 2 graphs, one for HPI and one for unemployment, over time and showing all 4 scenarios. Additionally, determine the date when the actual historical data ends and forecast starts (point when the series diverge across different scenarios). Which scenario implies severe economic conditions based on the forecast?__

# In[148]:


# You might need to convert 'Date' column to a datetime format if it's not already
df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])

# Create separate dataframes for HPI and Unemployment
df_hpi = df_transformed[['Date', 'HPI', 'Scenario']]
df_unemployment = df_transformed[['Date', 'Unemployment', 'Scenario']]

# Create a figure and axes for HPI graph
plt.figure(figsize=(10, 6))
plt.title('HPI Over Time for Different Scenarios')
for scenario in df_hpi['Scenario'].unique():
    scenario_data = df_hpi[df_hpi['Scenario'] == scenario]
    plt.plot(scenario_data['Date'], scenario_data['HPI'], label=scenario)
plt.xlabel('Date')
plt.ylabel('HPI')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()


# In[149]:


# Create a figure and axes for Unemployment graph
plt.figure(figsize=(10, 6))
plt.title('Unemployment Over Time for Different Scenarios')
for scenario in df_unemployment['Scenario'].unique():
    scenario_data = df_unemployment[df_unemployment['Scenario'] == scenario]
    plt.plot(scenario_data['Date'], scenario_data['Unemployment'], label=scenario)
plt.xlabel('Date')
plt.ylabel('Unemployment')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Find the date when historical data ends and forecast starts
historical_end_date = df_hpi['Date'].max()
forecast_start_date = historical_end_date + pd.DateOffset(months=3)

# Print the dates
print(f"Historical Data Ends: {historical_end_date}")
print(f"Forecast Starts: {forecast_start_date}")

# Show the graphs
plt.show()


# ## Problem 3

# For this exercise no coding is required.

# 1. __What is the difference between supervised and unsupervised machine learning? Which one would you use for client churn analysis?__

# - Supervised Learning: In supervised learning, the algorithm is trained on a labeled dataset, where the input data is paired with the corresponding correct output. The goal is to learn a mapping from inputs to outputs so that the algorithm can make accurate predictions on new, unseen data.
# - Unsupervised Learning: In unsupervised learning, the algorithm is given unlabeled data and aims to find patterns, relationships, or structures within the data without explicit guidance on what the correct output should be. Clustering and dimensionality reduction are common tasks in unsupervised learning.
# 
# For client churn analysis, supervised learning would be more appropriate. This is because churn analysis typically involves predicting whether a customer will churn or not (a binary classification problem), and you would have historical data with labeled examples of churn and non-churn cases.

# 2. __What are the most known ensemble algorithms?__

# Some of the most popular ensemble algorithms include:
# 
# - Random Forest: Builds multiple decision trees and combines their predictions.
# - Gradient Boosting (e.g., XGBoost, LightGBM, AdaBoost): Builds trees sequentially, each correcting the errors of the previous one.
# - Voting (Hard or Soft): Combines predictions from multiple models by majority voting or using weighted averages.
# - Bagging (Bootstrap Aggregating): Trains multiple models on different subsets of the data and averages their predictions.

# 3. __What is the difference between Type I and Type II error?__

# - Type I Error (False Positive): This occurs when a null hypothesis that is actually true is rejected. In other words, it's a "false alarm," where you detect an effect that doesn't exist.
# - Type II Error (False Negative): This occurs when a null hypothesis that is actually false is not rejected. In other words, you miss detecting a real effect.

# 4. __Explain the difference between L1 and L2 regularization methods.__

# Both thechniques are used to prevent overfiting.
# 
# - L1 Regularization (Lasso): Adds the absolute values of the coefficients to the loss function. It can lead to sparse solutions by driving some coefficients to exactly zero, effectively performing feature selection.
# - L2 Regularization (Ridge): Adds the squared values of the coefficients to the loss function. It encourages small values for all coefficients and is effective for reducing multicollinearity.

# 5. __Given the statistics for 2 classification models on Test dataset in the table below__
# 
#  - Which model predicts actual positives better?
#  - Which model is best and why, given the following information?

# Statistic | Model 1 | Model 2
# :---|:---:|:---:
# AUC ROC | 0.71 | 0.72
# Accuracy | 0.78 | 0.81
# Recall | 0.55 | 0.53
# Precision | 0.59 | 0.68
# F1-Score | 0.57 | 0.59

# - Model 2 predicts actual positives better based on higher recall (0.53 vs. 0.55).
# - In terms of overall performance, Model 2 is better because it has a slightly higher AUC ROC, accuracy, precision, and F1-score compared to Model 1. However, the differences between the two models are relatively small, so the choice could also depend on other factors such as interpretability, computational complexity, and business requirements.

# 6. __Given the confusion matrix below:__
#  - Calculate accuracy, recall and precision.
#  - Which number corresponds to Type I error?

# In[135]:


from IPython.display import HTML, display
import tabulate
table = [["","","Actual","Actual"],
         ["","",1,0],
         ["Predicted",1,350,25],
         ["Predicted",0, 10,210]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# __Answer: Although the table did not load, 25 is the type I error (False Positive)__
