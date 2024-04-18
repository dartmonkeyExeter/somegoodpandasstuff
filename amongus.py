import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from itertools import permutations
from scipy import stats
import numpy as np

pd.set_option('display.max_columns', 85)
pd.set_option('display.max_rows', 85)

api_key = 'f947faa37985758fc1ffe965'
url = 'https://v6.exchangerate-api.com/v6/f947faa37985758fc1ffe965/latest/GBP'

params = {
    'base': 'GBP'
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.get(url, headers=headers, params=params)
data = response.json()
#print(data)

convert_to_gbp_dict = data['conversion_rates']

df = pd.read_csv('survey_results_public.csv')
df = df.dropna(subset=['Currency'], axis=0)
df = df.dropna(subset=['CompTotal'], axis=0)

#print(df.shape)
#print(df.dtypes)

def convert_to_usd(row):
    if row['CompTotal'] > 0:
        currency_code = row['Currency'][:3]
        return row['CompTotal'] / convert_to_gbp_dict[currency_code]
    else:
        return row['CompTotal']

df['ConvertedToGBP'] = df.apply(convert_to_usd, axis=1)
#print(df[['Currency', 'CompTotal', 'ConvertedToGBP']])

df.drop(df[df['ConvertedToGBP'] > 20000000].index, inplace=True)

mean_converted_values = df.groupby('Country')['ConvertedToGBP'].mean()
mean_converted_values = mean_converted_values.sort_values(ascending=False)

#print(mean_converted_values.head(10))

plt.bar(mean_converted_values.index, mean_converted_values.values, width=0.9)
plt.scatter(df['Country'], df['ConvertedToGBP'], color='red', alpha=0.2, cmap='viridis', s=1)
plt.xlabel('Country')
plt.ylabel('Mean GBP Income')
plt.xticks(rotation='vertical')
plt.title('Mean GBP Income for Each Country')
plt.show()
plt.clf()

clearing_outliers = df[df['ConvertedToGBP'] < 300000]
plt.hist(clearing_outliers['ConvertedToGBP'], bins=50, alpha=0.75, color='blue')
plt.xlabel('Income in GBP')
plt.ylabel('Frequency')
plt.title('Distribution of Income in GBP')
plt.show()
plt.clf()

# grouped data analysis
#print(df.groupby('EdLevel')['ConvertedToGBP'].mean().sort_values(ascending=False))

more_than_5_years = df
more_than_5_years['YearsCode'] = pd.to_numeric(more_than_5_years['YearsCode'], errors='coerce')
more_than_5_years = more_than_5_years[more_than_5_years['YearsCode'] >= 5]
#print(more_than_5_years.groupby('YearsCode')['ConvertedToGBP'].mean().sort_values(ascending=False))

clearing_outliers = df[df['ConvertedToGBP'] < 300000]
plt.scatter(clearing_outliers['YearsCode'], clearing_outliers['ConvertedToGBP'], cmap="inferno", alpha=0.1, s=1)
plt.xlabel('Years of Coding Experience')
plt.ylabel('Income in GBP')
plt.title('Income in GBP by Years of Coding Experience')
plt.show()
plt.clf()

# time for correlation haha lol no correlation here but still fun to do it anyway

# trying to find correlation between categorics now

corr_df = df[['YearsCode', 'ConvertedToGBP', 'EdLevel', 'WorkExp', 'Age']]

ed_level_order = df['EdLevel'].unique().astype(str)
age_order = df['Age'].unique().astype(str)

corr_df['EdLevel'] = pd.Categorical(corr_df['EdLevel'], categories=ed_level_order, ordered=True)
corr_df['Age'] = pd.Categorical(corr_df['Age'], categories=age_order, ordered=True)

# now factorize the categorics
corr_df['EdLevel'] = corr_df['EdLevel'].factorize()[0]
corr_df['Age'] = corr_df['Age'].factorize()[0]

corr_mat = corr_df.corr(method='kendall')

sns.heatmap(corr_mat, annot=True)
plt.show()
plt.clf()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select relevant features
features = ['YearsCode', 'WorkExp']  # Add more features if needed

# Prepare the data
data = df.dropna(subset=features + ['ConvertedToGBP'], how='any')
X = data[features]
y = data['ConvertedToGBP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict GBP income for new data
new_data = pd.DataFrame({'YearsCode': [20], 'WorkExp': [15]})  # Example new data
predicted_income = model.predict(new_data)
print("Predicted GBP Income:", predicted_income[0])