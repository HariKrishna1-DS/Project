# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:34:30 2024

@author: sriha
"""
'''                         1.  IMPORTING DATASSETS'''

import pandas as pd
import matplotlib.pyplot as plt



data=pd.read_csv("C:\\Users\\sriha\\OneDrive\\Desktop\\pb excel\\covid internshipSudio.csv")
df=pd.DataFrame(data)
#print(df)


'''                     2. High level of Understanding'''
#FIND NUMBER OF ROWS AND COLUMNS IN DATASET

rows,columns = df.shape
print(df,"Number of Rows:{rows}")
print(df,"number of columns:{columns}")

#      DATAYPES OF COLUMNS
print("datatypes of columns \n",df.dtypes)

#    INFO  AND   DESCRIBE IN THE DATAFRAME
print("DESC OF  STASTICAL DATAFRAMES \n",df.describe())


'''                  3.    LOW LEVEL OF UNDERSTANDING'''

#        A. Count of unique values in the 'location' column
unique_locations_count = df['location'].nunique()
print(f'The number of unique values in the location column is: {unique_locations_count}')

#           B.  Calculate the frequency of each continent
continent_counts = df['continent'].value_counts()

# Find the continent with the maximum frequency
most_frequent_continent = continent_counts.idxmax()
max_frequency = continent_counts.max()
print(f'The continent with the maximum frequency is: {most_frequent_continent}')
print(f'The maximum frequency is: {max_frequency}')

#    C. FINDING MAXIMUM & MEAN VALUES IN 'TOTAL_CASES'.

# Find the maximum value in the 'total_cases' column
max_total_cases = df['total_cases'].max()
print(f'The maximum value in total_cases is: {max_total_cases}')

# Find the mean value in the 'total_cases' column
mean_total_cases = df['total_cases'].mean()
print(f'The mean value in total_cases is: {mean_total_cases}')


#    D.  FINDING  25% ,50%, &  75% QUTAERILE VALUES IN TOTAL_DEATHS

# Calculate quartile values for the 'total_deaths' column
quartiles = df['total_deaths'].quantile([0.25, 0.50, 0.75])

# Display the quartile values
print("25th percentile in total_deaths (Q1):", quartiles[0.25])
print("50th percentile in total_deaths (Q2):", quartiles[0.50])
print("75th percentile in total_deaths (Q3):", quartiles[0.75])


#   E.  FINDING WHICH CONTINENT HAS MAXIMUM  'HUMAN_DEVELOPMENT_INDEX'.


# Find the continent with the maximum 'human_development_index'
max_hdi_continent = df.loc[df['human_development_index'].idxmax()]['continent']
print(f'The continent with the maximum human development index is: {max_hdi_continent}')


#   F. FINDING MANIMUM 'GDP_PER_CAPITA'
min_gdp_continent = df.loc[df['gdp_per_capita'].idxmin()]['continent'] 
print(f'The continent with the minimum GDP per capita is: {min_gdp_continent}')





'''             4. FILTER THE DATAFRAME WITH ONLY THE SPCIFIED COLUMNS'''

columns_to_keep = ['continent', 'location', 'date', 'total_cases', 'total_deaths', 'gdp_per_capita', 'human_development_index'] 
df_filtered = df[columns_to_keep] 
# Update the original DataFrame with the filtered one
 
df = df_filtered
 # Display the updated DataFrame
print(df.head())

'''             5. DATA CLEANING'''

# A. REMOVE ALL DUPLICATE OBSERVATIONS

df_cleaned=df.drop_duplicates()
df=df_cleaned
print(df.head())

#B. FINDING MISSING VALUES IN ALL COLUMNS

missing_values=df.isnull().sum()
print("missing values in all coumns \n",missing_values)


#  C. REMOVE ALL OBSERVATIONS WHERE THE CONTINENT  COLUMN VALUES  IS MISSING

df_cleaned = df.dropna(subset=['continent']) 
 #  TIP : USING SUBSET PARAMETER  IN DROPNA
df = df_cleaned 
print(df.head())


# D.  FILL ALL MISSING VALUES IN "0".

df_filled=df.fillna(0)
df=df_filled
print(df.head())





'''          6.   DATA TIME FORMAT'''

#  A.   CONVERT THE 'DATE' COLUMN TO DATETIME FORMAT
'''
df['date'] = pd.to_datetime(df['date'])
print(df.head())
'''

#  B.  CONVERT THE DATE COLUMN TO DATETIME FORMAT
'''
df['date'] = pd.to_datetime(df['date'])
 # Extract month data and create a new column 'month'
df['month'] = df['date'].dt.month
print(df.head())
'''




'''           7. DATA AGGREGATION  '''

#    A. Group by 'continent' and find the max value in all columns

df_max = df.groupby('continent').max().reset_index() 
 # Display the resulting DataFrame
print("max value in all columns using groupby function oncontinent  /n",df_max)


# B.   Group by 'continent' and find the max value in all column

df_groupby = df.groupby('continent').max().reset_index()
print(" Group by 'continent' and find the max value in all columns  /n ",
      df_groupby.head())








'''             8. FEATURE ENGINEERNING       '''

# Create the new feature 'total_deaths_to_total_cases' 
df['total_deaths_to_total_cases'] = df['total_deaths'] / df['total_cases']
print(" Create the new feature 'total_deaths_to_total_case",df.head())








'''                9. data visualization '''

'''
#  A.     Plotting the distribution of 'gdp_per_capita'
plt.figure(figsize=(10, 6)) 
sns.histplot(df['gdp_per_capita'], bins=30, kde=True)
plt.title('Distribution of GDP per Capita')
plt.xlabel('GDP per Capita')
plt.ylabel('Frequency') 
plt.grid(True) 
plt.show()'''


# B.   PLOT A SCATTER PLOT OF 'TOTAL_CASES' & 'GDP_PER_CAPIA

x=["gdp_per_capita"]
y=["total_cases"]

plt.figure(figsize=(10,6))
plt.scatter(x, y,data=df)
plt.xlabel="GDP_PER_CAPITA"
plt.ylabel="TOTAL_CASES"
plt.title("total_cases  &   gdp_per_capita")
plt.grid(True)
plt.show()


#    C.Plot a bar plot of 'continent' column with 'total_cases' .
    #     Tip : using kind='bar' in seaborn catplot
    

'''
sns.catplot(
    data=df, 
    a='continent', 
    b='total_cases', 
    kind='bar',
    height=6, 
    aspect=1.5
)

# Set plot titles and labels
plt.title('Total Cases by Continent')
plt.xlabel('Continent')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)

# Display the plot
plt.show()
'''


'''        10.    save the file      '''


# Save the df_groupby DataFrame to a CSV file
df_groupby.to_csv('df_groupby.csv', index=False) 
 # Confirm save
print("DataFrame saved as 'df_groupby.csv'")