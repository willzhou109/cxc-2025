# -*- coding: utf-8 -*-

"""# Indicator Analysis"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

file_path = 'normalized_dataset.csv'
data = pd.read_csv(file_path)

# Changing the data into a wide format
wide_data = data.pivot(index='Country Name', columns='Indicator Name', values='Normalized_Row_Average')
wide_data.reset_index(inplace=True)

# Cleaning the wide dataset
print("Missing values before imputation:")
print(wide_data.isnull().sum())

indicator_columns = wide_data.columns.drop('Country Name')

wide_data[indicator_columns] = wide_data[indicator_columns].apply(pd.to_numeric, errors='coerce')

wide_data[indicator_columns] = wide_data[indicator_columns].apply(lambda row: row.fillna(row.mean()), axis=1)

print("\nMissing values after imputation:")
print(wide_data.isnull().sum())

output_file_path = 'cleaned_wide_data.csv'
wide_data.to_csv(output_file_path, index=False)

print(wide_data.head())

indicators_file_path = 'cleaned_wide_data.csv'
mpi_file_path = 'mpi_results.csv'
indicators_data = pd.read_csv(indicators_file_path)
mpi_data = pd.read_csv(mpi_file_path)

# Merging the mpi dataset and the wide dataset
combined_data = pd.merge(indicators_data, mpi_data, on='Country Name', how='left')

# Setting up Random Forest
X = combined_data.drop(columns=['Country Name', 'MPI'])
y = combined_data['MPI']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Analyzing feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)
print("\nFeature Importances:")
print(importances_sorted)

output_file_path = 'combined_data.csv'
combined_data.to_csv(output_file_path, index=True)
print(f"\nCombined data has been saved to {output_file_path}")
