import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('Sample Media Spend Data.xlsx')

df = df[df['Division'] == 'A']

print(df.shape)
print(df.head())
print(df.columns)
print(df.isnull().sum())

df['Calendar_Week'] = pd.to_datetime(df['Calendar_Week'])
print(df.dtypes)
print(df.head())
print(df.dtypes)
print(df['Calendar_Week'].min())
print(df['Calendar_Week'].max())

# At this point I have found out how many rows and columns are in my data set-[113,10]
# using min and max dates; I can find the time period of the data - 2018-01-09 to 2020-11-01
# my target variable is sales - I am trying to predict sales based on the marketing channels
# my marketing channels is all the impressions plus paid and organic views,
# using line 9, we can find out if there are any missing values - there are 0

df_sorted = df.sort_values('Calendar_Week')
df_sorted.plot(x='Calendar_Week', y='Sales', figsize=(12, 4), title='Sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
plt.close()

spend_cols = ['Paid_Views', 'Organic_Views', 'Google_Impressions',
              'Email_Impressions', 'Facebook_Impressions', 'Affiliate_Impressions']

df_sorted.plot(x='Calendar_Week', y=spend_cols, figsize=(12, 6),
               title='Marketing Channels over time')
plt.xlabel('Date')
plt.ylabel('Views / Impressions')
plt.legend(loc='best')
plt.show()
plt.close()

# at this point the graphs are showing that Google impression has a very similar pattern to sales
# google impression seems to be the highest values
# paid views, organic views, and affiliate impressions all seem to be around 0 throughout the period
# as the views increases so do the sales showing a positive correlation

corr_matrix = df_sorted[spend_cols + ['Sales']].corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
plt.close()

# now we can see the correlation, and from this we can see google impression has the strongest cor to sales: 0.71
# affiliate impressions have a negative cor with sales, meaning when affiliate impressions increase, sales fall
# paid and organic views, as well as facebook and Google impressions both have a cor of around 0.7 suggestions strong correlation making them multicollinearity (channels that cor with each other)

# Define predictors and target
X = df_sorted[spend_cols]
X = sm.add_constant(X)  # adds an intercept term (the baseline/starting point)
y = df_sorted['Sales']

# Fit the model
baseline_model = sm.OLS(y, X).fit()

# Print the summary
print(baseline_model.summary())

# r squared tells explains the variation- 66.5% of
# all channels are sig except organic views and paid views as p> 0.05
# google impression has the largest effect on sales as it has the largest coefficient
# because they are both correlated to each other it makes both of them unreliable

def apply_adstock(spend_series, decay_rate):
    adstocked = spend_series.copy().astype(float)  # convert to float first
    for t in range(1, len(adstocked)):
        adstocked.iloc[t] = spend_series.iloc[t] + decay_rate * adstocked.iloc[t-1]
    return adstocked

df_sorted['Google_Impressions_stocked'] = apply_adstock(df_sorted['Google_Impressions'], decay_rate=0.5)
plt.figure(figsize=(12, 5))
plt.plot(df_sorted['Calendar_Week'], df_sorted['Google_Impressions'], label='Raw', marker='o', alpha=0.7)
plt.plot(df_sorted['Calendar_Week'], df_sorted['Google_Impressions_stocked'], label='Adstocked (decay=0.5)', marker='s', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Impressions')
plt.title('Google Impressions: Raw vs Adstocked')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

for col in spend_cols:
    df_sorted[col + '_stocked'] = apply_adstock(df_sorted[col], decay_rate=0.5)
print(df_sorted.columns)

# Adstock Transformation Complete.I applied adstock with decay of 0.5 to all marketing channels. the adstock takes into account the lingering effect of marketing

def find_best_decay_rate(spend_series, sales_series):
    """
    Test different decay rates and find which one gives the best R-squared.
    """
    optimal_decay = 0.1
    optimal_r2 = 0.0

    for decay_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        adstocked_data = apply_adstock(spend_series, decay_rate)
        x_with_const = sm.add_constant(adstocked_data)
        regression_model = sm.OLS(sales_series, x_with_const).fit()

        current_r2 = float(regression_model.rsquared)
        if current_r2 > optimal_r2:
            optimal_r2 = current_r2
            optimal_decay = decay_rate

    return optimal_decay, optimal_r2

best_decays = {}
for col in spend_cols:
    best_decay, r2 = find_best_decay_rate(df_sorted[col], df_sorted['Sales'])
    best_decays[col] = (best_decay, r2)
    print(f"{col}: decay_rate={best_decay}, R-squared={r2:.4f}")

# I tested decay rates from 0.1 to 0.9 for each channel independently. the decay rate that gave the highest r^2 means that is the most fitting rate for that channel. results show a Google impression has decay of 0.1 so they have quick effects, whereas organic views have decay of 0.9, meaning the lingering effect is longer

# Create a list of final (adstocked + log-transformed) columns
adstocked_cols = [col + '_stocked' for col in spend_cols]

# Apply log transformation to each adstocked column
for col in adstocked_cols:
    df_sorted[col.replace('_stocked', '') + '_final'] = np.log(df_sorted[col] + 1)

print(df_sorted.columns)
# now diminishing returns has been applied using a log transformation to each adstock. this means that each additional unit of spending is less impactful than the previous one

# Define predictors using the final transformed columns
final_cols = [col + '_final' for col in spend_cols]
X_final = df_sorted[final_cols]
X_final = sm.add_constant(X_final)
y = df_sorted['Sales']

# Fit the final model
final_model = sm.OLS(y, X_final).fit()

# Print the summary
print(final_model.summary())

# Use just the adstocked columns (no log transformation)
adstocked_cols_list = [col + '_stocked' for col in spend_cols]
X_adstocked = df_sorted[adstocked_cols_list]
X_adstocked = sm.add_constant(X_adstocked)
y = df_sorted['Sales']

# Fit the model with adstocked (but not log-transformed) columns
adstocked_model = sm.OLS(y, X_adstocked).fit()
print(adstocked_model.summary())

# I've now tested three approaches: 1) baseline - R^2 = 0.665, 2) Adstocked only: R² = 0.567, 3. Adstocked + Log-transformed: R² = 0.503
# baseline has the highest r^2 showing it perforates the best with raw spend. this suggests that the data is most likely immediate rather than having long carryover effects,
# therefore, use baseline as a final model

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Use the raw speed columns (from the baseline model)
X_for_vif = df_sorted[spend_cols]

# Calculate VIF for each column - VIF tells us How much does multicollinearity with other variables inflate the uncertainty of this variable's coefficient?
vif_data = pd.DataFrame()
vif_data["Feature"] = X_for_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_for_vif.values, i) for i in range(len(X_for_vif.columns))]

print(vif_data)

# VIF shows that email impressions (9.31) and affiliate impressions (7.11) have a high multicollinearity with other channels. this means their individual coefficients are less reliable. This means we should be cautious about using email and affiliate coefficients

# now we have to turn the statistics into business insights.
# 1) calc total estimated contribution
# 2) calc rough ROI par channel
# 3) rank ROI and create visual representation

# Get the coefficients from your baseline model
results = baseline_model.params

# Calculate contributions for each channel
contributions = {}
for i, channel in enumerate(spend_cols):
    coef = results.iloc[i + 1]  # Use .iloc to access by position
    total_activity = df_sorted[channel].sum()
    contribution = coef * total_activity
    contributions[channel] = contribution

# Create a dataframe
contribution_df = pd.DataFrame(list(contributions.items()), columns=['Channel', 'Contribution'])
contribution_df = contribution_df.sort_values('Contribution', ascending=False)

print(contribution_df)

# Create a bar chart
import matplotlib.pyplot as plt
contribution_df.plot(kind='bar', x='Channel', y='Contribution', legend=False, figsize=(10, 5))
plt.title('Estimated Sales Contribution by Channel')
plt.ylabel('Contribution to Sales')
plt.xlabel('Marketing Channel')
plt.tight_layout()
plt.show()
plt.close()

# Calculate ROI: contribution per unit of channel activity
roi_df = contribution_df.copy()
roi_df['Total_Activity'] = [df_sorted[channel].sum() for channel in roi_df['Channel']]
roi_df['ROI_per_Unit'] = roi_df['Contribution'] / roi_df['Total_Activity']
roi_df = roi_df.sort_values('ROI_per_Unit', ascending=False)

print(roi_df[['Channel', 'Contribution', 'Total_Activity', 'ROI_per_Unit']])

"""
# Channel ROI Analysis & Recommendation

Based on the baseline regression model (R² = 0.665), here are the key findings:

**By Total Contribution to Sales:**
1. Email Impressions: $4.2M contribution
2. Google Impressions: $2.8M contribution
3. Facebook Impressions: $2.6M contribution

**By ROI Efficiency (contribution per unit):**
1. Organic Views: 0.21 (most efficient)
2. Facebook: 0.14
3. Email: 0.10

**Concerning Findings:**
- Paid Views (-0.54 ROI) and Affiliate (-6.32 ROI) are both negative predictors
- This suggests these channels are either inefficient or the model is capturing 
  a negative correlation that warrants investigation

**Recommendation:**
Prioritize Organic Views for efficiency, but maintain Email/Google/Facebook for volume.
Consider investigating why Paid Views and Affiliate Impressions show negative returns —
this could indicate data quality issues, poor campaign targeting, or genuine inefficiency.
"""

