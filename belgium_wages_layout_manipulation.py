# DESCRIPTION:      This script modifies file layout from cross-sectional to horizontal. It was created for work purposes to 
#                   automate data entry process. This script loops through all years filtering data for each year and saving the result to a list.
#                   Each element from the list is a dataframe that contains one year worth of data. The elements of the loop
#                   are then appended to a single dataframe and sorted to a format manageable by the XLS 2D data reader used in Moody's.  
# WRITTEN:          Maciej Milanowski, 10/17/2022

import pandas as pd

# Get the last year from powershell (here excluded)
year = '2022'
last_year = int(year)

# Read the file
data = pd.read_excel("https://statbel.fgov.be/sites/default/files/files/documents/Conjunctuur/4.3%20Loonmassa/WAGES_2015_EN.xls")

# Get the first year from data
start_obs = data['YEAR']
first_year = start_obs.min()

# Sheet 1
df1 = data[data['YEAR'] >= last_year]
df1 = df1.rename(columns={'Q_1': year+'Q_1','Q_2': year+'Q_2','Q_3': year+'Q_3','Q_4': year+'Q_4'})
df1 = df1.reset_index(drop=False)

# Loop to filter each year from the file and 
df_list = []

for i in range(first_year, last_year):
    df_i = data[data['YEAR'] == i]
    df_i = df_i.rename(columns={'Q_1': str(i)+'Q_1','Q_2': str(i)+'Q_2','Q_3': str(i)+'Q_3','Q_4': str(i)+'Q_4'})
    df_i = df_i.reset_index(drop=False)
    df_i = df_i.iloc[:,-4:]
    df_list.append(df_i)
    
df_new = pd.concat(df_list)

# Filtering out NaN records in each column
df_new = df_new.apply(lambda x: pd.Series(x.dropna().values))

# Merge the last year with the rest of years
df_merged = pd.merge(left=df1, right=df_new, left_index=True, right_index=True)

# Sorting the new dataframe
df_sorted = df_merged.iloc[:,5:]
df_sorted = df_sorted.reindex(sorted(df_sorted.columns), axis=1)

df_labels = df_merged.iloc[:,:5]
df_export = pd.merge(left=df_labels, right=df_sorted, left_index=True, right_index=True)

#Exporting for reading
with pd.ExcelWriter("horizontal_layout.xlsx") as writer:
    df_export.to_excel(writer, index=False)