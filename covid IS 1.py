
# a. Find the number of rows and columns in the dataset
rows, columns = df.shape
print(f'Number of rows: {rows}')
print(f'Number of columns: {columns}')

# b. Data types of columns
print("\nData types of columns:\n", df.dtypes)

# c. Info of the data in the DataFrame
print("\nInfo of the DataFrame:")
print(df.info())

# c. Describe the data in the DataFrame
print("\nDescriptive statistics of the DataFrame:")
print(df.describe())
# 3.
# Count of unique values in the 'location' column
unique_locations_count = df['location'].nunique()

print(f'The number of unique values in the location column is: {unique_locations_count}')
