#!/usr/bin/env python
# coding: utf-8

# # Which one is a better plan?
# 
# You work as an analyst for the telecom operator Megaline. The company offers its clients two prepaid plans, Surf and Ultimate. The commercial department wants to know which of the plans brings in more revenue in order to adjust the advertising budget.
# 
# You are going to carry out a preliminary analysis of the plans based on a relatively small client selection. You'll have the data on 500 Megaline clients: who the clients are, where they're from, which plan they use, and the number of calls they made and text messages they sent in 2018. Your job is to analyze the clients' behavior and determine which prepaid plan brings in more revenue.

# ## Initialization

# 

# In[1]:


# Loading all the libraries
import pandas as pd
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
import math
import seaborn as sns


# ## Load data

# 

# In[2]:


# Load the data files into different DataFrames
calls = pd.read_csv('/datasets/megaline_calls.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
plans = pd.read_csv('/datasets/megaline_plans.csv')
users = pd.read_csv('/datasets/megaline_users.csv')


# ## Plans

# In[3]:


# Print the general/summary information about the plans' DataFrame
plans.info()


# In[4]:


# Print a sample of data for plans
display(plans.tail())


# There is a column designated for usd_per_gb but not not the amount of gigabytes that are included in the plan. May cause confusion, I would make a seperate column designates for gb_per_month_included to help. 

#  

# ## Fix data

# plans['gb_per_month_included'] = plans['mb_per_month_included'] / 1024
# display(plans.head())

# In[5]:


plans['gb_per_month_included'] = plans['mb_per_month_included'] / 1024
display(plans.head())


# ## Enrich data

# [Add additional factors to the data if you believe they might be useful.]

# In[6]:


#No other issues seen currently


# 

# In[7]:


# Print the general/summary information about the users' DataFrame
users.info()


# In[8]:


# Print a sample of data for users
display(users)
users.isna().sum()


# Looking at the data I was initially concerned about the Non-Null values but looks like there are some applicable dates, with 500 listed and there are 466 Non-Null values out of the 500 rows. This information may be important later so will not modify this at this time.
# Only other note would be the dates for the reg_date and churn_date should be changed to datetime objects from its current objects status.

#  

# ## Fix Data

# [Fix obvious issues with the data given the initial observations.]

# In[9]:


# Convert reg_date to datetime format
users['reg_date'] = pd.to_datetime(users['reg_date'], format='%Y-%m-%d')
users['churn_date'] = pd.to_datetime(users['churn_date'], format='%Y-%m-%d')
users.info()


# ## Enrich Data

# [Add additional factors to the data if you believe they might be useful.]

# # Calls

# In[10]:


# Print the general/summary information about the calls' DataFrame
calls.info()


# In[11]:


# Print a sample of data for calls
display(calls.sample(n = 10, replace = True))


# Did not notice any missing values. The call_date values should be modified to datetime data type. It appears that the dates are randomized and not in order, having it organized will help keep data more organized. 

#  

# ## Fix data

# 

# In[12]:


# Converting call_date to datetime format
calls['call_date'] = pd.to_datetime(calls['call_date'], format='%Y-%m-%d')
# Pull month of the call data to a new column
calls['month'] = calls['call_date'].dt.month_name()


# ## Enrich data

# To keep data more streamlined I would think keeping the "duration" column rounded to whole numbers to ensure clear visualization.
# Later on we need to look at data in a monthly time frame. By making a column for the month to help track minutes per month.

# In[13]:


# Rounding the values in the 'duration' column to the next highest whole number.
calls['duration'] = np.ceil(calls['duration'])
display(calls.head(10))
# Want to get rid of decimal place in the "duration" column, make the data type to integer to reduce confusion.
calls['duration'] = calls['duration'].astype('int')
calls.head(10)
# Pull month of the call data to a new column
calls['month'] = calls['call_date'].dt.month_name()


# ## Messages

# In[14]:


# Print the general/summary information about the messages' DataFrame

messages.info()


# In[15]:


# Print a sample of data for messages
display(messages.sample( n =5))


# The message_date column values are strings and should be changed to datetime data type, to keep uniformity. 

#  

# ## Fix data

# [Fix obvious issues with the data given the initial observations.]

# In[16]:


# Change message_date to datetime format
messages['message_date'] = pd.to_datetime(messages['message_date'], format='%Y-%m-%d')
display(messages.info())


# ## Enrich data

# Would make a column for month so that the messages can stored, to be able to look up later.

# In[17]:


# Add month column for message_date
messages['month'] = messages['message_date'].dt.month_name()
messages.info()


# # Internet

# In[18]:


# Print the general/summary information about the internet DataFrame
internet.info()


# In[19]:


# Print a sample of data for the internet traffic
internet.sample(n = 10)


# Only issue that I see is the session_date column should be changed to the datetime data type that has been previously done.

#  

# ### Fix data

# [Fix obvious issues with the data given the initial observations.]

# In[20]:


# Change session_date to datetime format
internet['session_date'] = pd.to_datetime(internet['session_date'], format='%Y-%m-%d')


# ### Enrich data

# Adding an extra column for the month (session_date) to help in analyzing data later.

# In[21]:


# Add month column for session_date
internet['month'] = internet['session_date'].dt.month_name()


# ## Study plan conditions

# [It is critical to understand how the plans work, how users are charged based on their plan subscription. So, we suggest printing out the plan information to view their conditions once again.]

# In[22]:


# Print out the plan conditions and make sure they are clear for you
plans.head()


# ## Aggregate data per user
# 
# [Now, as the data is clean, aggregate data per user per period in order to have just one record per user per period. It should ease the further analysis a lot.]

# In[23]:


# Calculate the number of calls made by each user per month. Save the result.
calls_per_month = calls.groupby(['user_id', 'month'])['duration'].count().reset_index()
display(calls_per_month.head())


# In[24]:


# Calculate the amount of minutes spent by each user per month. This is made easier due to rounding up the minutes earlier. Save the result.
# Results will be stored in minutes_per_month variable
minutes_per_month = calls.groupby(['user_id', 'month'])['duration'].sum().reset_index()
display(minutes_per_month.head())


# In[25]:


# Calculate the number of messages sent by each user per month. Save the result.
# Stored in message_per_month
messages_per_month = messages.groupby(['user_id', 'month'])['id'].count().reset_index()
display(messages_per_month.head())


# In[26]:


# Calculate the volume of internet traffic used by each user per month. Save the result.
# Info saved to internet_per_month variable
internet_per_month = internet.groupby(['user_id', 'month'])['mb_used'].sum().reset_index()
display(internet_per_month.head(10))


# [Put the aggregate data together into one DataFrame so that one record in it would represent what an unique user consumed in a given month.]

# In[27]:


# Merge the data for calls, minutes, messages, internet based on user_id and month
merged_data = calls_per_month.merge(right=minutes_per_month, on = ['user_id', 'month'], how = 'outer')
merged_data = merged_data.merge(right=messages_per_month, on = ['user_id', 'month'], how = 'outer')
merged_data = merged_data.merge(right=internet_per_month, on = ['user_id', 'month'], how = 'outer')

# Fill the NaN values with 0
merged_data = merged_data.fillna(0)

# rename the columns
merged_data.columns = ['user_id', 'month', 'calls', 'minutes', 'messages', 'mb_used']

# reset the index
merged_data = merged_data.reset_index(drop = True)

# display the merged data
display(merged_data.head(10))


# In[28]:


# Add the plan information
plan_info = users[['user_id', 'plan']]
plan_info.columns = ['user_id', 'plan_name']

merged_data = merged_data.merge(right=plan_info, on = 'user_id')

# Merge the plan data to the merged data
merged_data = merged_data.merge(right=plans, on = 'plan_name')

# sort the data by user_id
merged_data = merged_data.sort_values(by='user_id').reset_index(drop = True)

# display the merged data
display(merged_data.head(20))


# [Calculate the monthly revenue from each user (subtract the free package limit from the total number of calls, text messages, and data; multiply the result by the calling plan value; add the monthly charge depending on the calling plan). N.B. This might not be as trivial as just a couple of lines given the plan conditions! So, it's okay to spend some time on it.]

# In[29]:


# Calculate the monthly revenue for each user
def calc_revenue(row):
    """Function to calculate the monthly revenue for each user"""
    # User's usage
    minutes = row['minutes']
    messages = row['messages']
    # for internet usage, change from megabytes to gigabytes and round up to the next highest whole gigabyte
    internet = math.ceil(row['mb_used'] * (2**(-10)))
    
    # Plan limits
    minutes_limit = row['minutes_included']
    messages_limit = row['messages_included']
    # for internet usage, change from megabytes to gigabytes
    internet_limit = row['mb_per_month_included'] * (2**(-10))
    
    # Fee rates
    monthly_rate = row['usd_monthly_pay']
    minute_rate = row['usd_per_minute']
    message_rate = row['usd_per_message']
    internet_rate = row['usd_per_gb']
    
    # Initialize the revenue values and assign them to 0
    minutes_revenue = 0
    messages_revenue = 0
    internet_revenue = 0
    
    # Calculating overuse and total revenue
    if minutes > minutes_limit:
        minutes_revenue = (minutes - minutes_limit) * minute_rate
    if messages > messages_limit:
        messages_revenue = (messages - messages_limit) * message_rate
    if internet > internet_limit:
        internet_revenue = (internet - internet_limit) * internet_rate
        
    revenue = monthly_rate + minutes_revenue + messages_revenue + internet_revenue
    
    return revenue


# In[30]:


# Test For Calc_Revenue
test_rows = merged_data.loc[:10]
display(test_rows.apply(calc_revenue, axis = 1))


# In[31]:


# Add the Revenue function to merged data
merged_data['revenue'] = merged_data.apply(calc_revenue, axis = 1)
display(merged_data.head(10))


# ## Study user behaviour

# [Calculate some useful descriptive statistics for the aggregated and merged data, which typically reveal an overall picture captured by the data. Draw useful plots to help the understanding. Given that the main task is to compare the plans and decide on which one is more profitable, the statistics and the plots should be calculated on a per-plan basis.]
# 
# [There are relevant hints in the comments for Calls but they are not provided for Messages and Internet though the principle of statistical study is the same for them as for Calls.]

# ## Calls

# In[32]:


# Compare average duration of calls per each plan per each distinct month. Plot a bar plat to visualize it.

# Get the data for the surf plan and plot the bar plot
surf_data = merged_data[merged_data['plan_name'] == 'surf']
surf_data = surf_data.groupby('month')['minutes'].mean().reset_index()
surf_data.columns = ['month', 'surf_minutes']

# Get the data for the ultimate plan and plot the bar plot
ultimate_data = merged_data[merged_data['plan_name'] == 'ultimate']
ultimate_data = ultimate_data.groupby('month')['minutes'].mean().reset_index()
ultimate_data.columns = ['month', 'ultimate_minutes']

# Merge data
merged_calls = surf_data.merge(right=ultimate_data, on = 'month', how = 'outer')

# sorting the index by the chronologically order of months
merged_calls.index = pd.CategoricalIndex(merged_calls['month'], categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
merged_calls = merged_calls.sort_index()



# Plot the bar plot
merged_calls.plot(x = 'month', y = ['surf_minutes', 'ultimate_minutes'], kind = 'bar', figsize = (15, 10))
plt.title('Average minutes per month for each plan')
plt.xlabel('Month')
plt.ylabel('Minutes')
plt.show()


# In[33]:


# Compare the number of minutes users of each plan require each month. Plot a histogram.

# Acquire the data for the surf plan and plot the histogram
surf_data = merged_data[merged_data['plan_name'] == 'surf']['minutes'].hist(bins=50, figsize = (15, 10), label = 'surf')

# Acquire the data for the ultimate plan and plot the histogram
ultimate_data = merged_data[merged_data['plan_name'] == 'ultimate']['minutes'].hist(bins=50, figsize = (15, 10), label = 'ultimate')

# Setting the title and labels
plt.title('Minutes per month for each plan')
plt.xlabel('Minutes')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# [Calculate the mean and the variable of the call duration to reason on whether users on the different plans have different behaviours for their calls.]

# In[34]:


# Calculate the mean and the variance of the monthly call duration

# Get the data for the surf plan
surf_data_calls = merged_data[merged_data['plan_name'] == 'surf'].groupby('user_id')['minutes'].mean()

# Get the data for the ultimate plan
ultimate_data_calls = merged_data[merged_data['plan_name'] == 'ultimate'].groupby('user_id')['minutes'].mean()

# Calculate the mean, variance, and standard deviation for the surf plan
surf_mean_calls = surf_data_calls.mean()
surf_var_calls = surf_data_calls.var()
surf_std_calls = np.std(surf_data_calls)

# Calculate the mean, variance, and standard deviation for the ultimate plan
ultimate_mean_calls = ultimate_data_calls.mean()
ultimate_var_calls = ultimate_data_calls.var()
ultimate_std_calls = np.std(ultimate_data_calls)

# Print the results
print(f'Mean for the surf plan: {round(surf_mean_calls,2)}')
print(f'Variance for the surf plan: {round(surf_var_calls,2)}')
print(f'Standard deviation for the surf plan: {round(surf_std_calls,2)}')
print(f'Mean for the ultimate plan: {round(ultimate_mean_calls,2)}')
print(f'Variance for the ultimate plan: {round(ultimate_var_calls,2)}')
print(f'Standard deviation for the ultimate plan: {round(ultimate_std_calls,2)}')


# In[35]:


# Plot a boxplot to visualize the distribution of the monthly call duration
sns.boxplot(x = 'plan_name', y = 'minutes', data = merged_data, palette = 'Set2',)
plt.title('Distribution of the monthly call duration')
plt.xlabel('Plan')
plt.ylabel('Minutes')
plt.show()


# The boxplots look very similiar. this helps to back up the calculations for the mean and standard deviations earlier. Clients from both plans seem to average 420 minutes call usage.

#  

# ## Messages

# In[36]:


# Compare the number of messages users of each plan tend to send each month

# Get the data for the surf plan 
surf_messages = round(merged_data[merged_data['plan_name'] == 'surf'].groupby('month')['messages'].mean(), 2)

# Get the data for the ultimate plan
ultimate_messages = round(merged_data[merged_data['plan_name'] == 'ultimate'].groupby('month')['messages'].mean(), 2)

# Merge the data 
merged_messages = surf_messages.to_frame().merge(right=ultimate_messages.to_frame(), on = 'month', how = 'outer')

# rename messages_x and messages_y to surf_messages and ultimate_messages
merged_messages.columns = ['surf_messages', 'ultimate_messages']

# sort the index by the chronologically order of months
merged_messages.index = pd.CategoricalIndex(merged_messages.index, categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
merged_messages.sort_index(inplace=True)



# Plot the bar plot
merged_messages.plot(kind = 'bar', figsize = (15, 10))
plt.title('Average messages per month for each plan')
plt.xlabel('Month')
plt.ylabel('Messages')
plt.show()


# In[37]:


# Compare the amount of messages sent by users per plan

# Get the data for the surf plan and plot the histogram
surf_data = merged_data[merged_data['plan_name'] == 'surf']['messages'].hist(bins=50, figsize = (15, 10), label = 'surf')

# Get the data for the ultimate plan and plot the histogram

ultimate_data = merged_data[merged_data['plan_name'] == 'ultimate']['messages'].hist(bins=50, figsize = (15, 10), label = 'ultimate')

# Set the title and labels
plt.title('Messages per month for each plan')
plt.xlabel('Messages')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[38]:


# Find the mean and the variance of the monthly messages sent

# Get the data for the surf plan
surf_data_messages = merged_data[merged_data['plan_name'] == 'surf'].groupby('user_id')['messages'].mean()

# Get the data for the ultimate plan
ultimate_data_messages = merged_data[merged_data['plan_name'] == 'ultimate'].groupby('user_id')['messages'].mean()

# Calculate the mean, variance, and standard deviation for the surf plan
surf_mean_messages = surf_data_messages.mean()
surf_var_messages = surf_data_messages.var()
surf_std_messages = np.std(surf_data_messages)

# Calculate the mean, variance, and standard deviation for the ultimate plan
ultimate_mean_messages = ultimate_data_messages.mean()
ultimate_var_messages = ultimate_data_messages.var()
ultimate_std_messages = np.std(ultimate_data_messages)

# Print the results
print(f'Mean for the surf plan: {round(surf_mean_messages,2)}')
print(f'Variance for the surf plan: {round(surf_var_messages,2)}')
print(f'Standard deviation for the surf plan: {round(surf_std_messages,2)}')
print(f'Mean for the ultimate plan: {round(ultimate_mean_messages,2)}')
print(f'Variance for the ultimate plan: {round(ultimate_var_messages,2)}')
print(f'Standard deviation for the ultimate plan: {round(ultimate_std_messages,2)}')


# The average number of messages are about 33 for the clients with the Surf plan and those under the Ultimate plan is 39 messages. Standard deviation are relatively the same with 31 or 32. We do see that there are a decent amount of clients in the Surf group that exceed 50 messages a month that are prepaid, This leads to extra revenue from those clients that exceed those 50 prepaid messages.

#  

# ## Internet

# In[39]:


# Compare the amount of internet traffic consumed by users per plan

# Get the data for the surf plan and plot the histogram
surf_data = merged_data[merged_data['plan_name'] == 'surf']['mb_used'].hist(bins=50, figsize = (15, 10), label = 'surf')

# Get the data for the ultimate plan and plot the histogram
ultimate_data = merged_data[merged_data['plan_name'] == 'ultimate']['mb_used'].hist(bins=50, figsize = (15, 10), label = 'ultimate')

# Set the title and labels
plt.title('Internet usage per month for each plan')
plt.xlabel('Internet Usage (MB)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[40]:


# Calculate the mean and the variance of the monthly internet usage
# Get the data for the surf plan
surf_data_internet = merged_data[merged_data['plan_name'] == 'surf'].groupby('user_id')['mb_used'].mean()

# Get the data for the ultimate plan
ultimate_data_internet = merged_data[merged_data['plan_name'] == 'ultimate'].groupby('user_id')['mb_used'].mean()

# Calculate the mean, variance, and standard deviation for the surf plan
surf_mean_internet = surf_data_internet.mean()/1000
surf_var_internet = surf_data_internet.var()/(1000**2)
surf_std_internet = np.std(surf_data_internet)/1000

# Calculate the mean, variance, and standard deviation for the ultimate plan
ultimate_mean_internet = ultimate_data_internet.mean()/1000
ultimate_var_internet = ultimate_data_internet.var()/(1000**2)
ultimate_std_internet = np.std(ultimate_data_internet)/1000


# Print the results
print(f'Mean for the surf plan: {round(surf_mean_internet,2)} gb')
print(f'Variance for the surf plan: {round(surf_var_internet,2)} gb')
print(f'Standard deviation for the surf plan: {round(surf_std_internet,2)} gb')
print(f'Mean for the ultimate plan: {round(ultimate_mean_internet,2)} gb')
print(f'Variance for the ultimate plan: {round(ultimate_var_internet,2)} gb')
print(f'Standard deviation for the ultimate plan: {round(ultimate_std_internet,2)} gb')


# In[41]:


# Plot a boxplot to show the distribution of the monthly internet usage
sns.boxplot(x = 'plan_name', y = 'mb_used', data = merged_data, palette = 'Set2')
plt.title('Distribution of the monthly internet usage')
plt.xlabel('Plan')
plt.ylabel('Internet Usage (MB)')
plt.show()


# Looking at the information, it looks like the average monthly internet usage for clients in the Surf plan exceed the plan's limit of 15GB, which leads to additional fee to monthly usage. Those in the Ultimate plan mostly stay within their 30GB range in internet data.

#  

# # Revenue

# [Likewise you have studied the user behaviour, statistically describe the revenue between the plans.]

# In[42]:


# Compare the revenue from users of each plan

# Get the data for the surf plan
surf_revenue = round(merged_data[merged_data['plan_name'] == 'surf'].groupby('month')['revenue'].mean(), 2)

# Get the data for the ultimate plan
ultimate_revenue = round(merged_data[merged_data['plan_name'] == 'ultimate'].groupby('month')['revenue'].mean(), 2)

# Merge the data
merged_revenue = surf_revenue.to_frame().merge(right=ultimate_revenue.to_frame(), on = 'month', how = 'outer')

# rename revenue_x and revenue_y to surf_revenue and ultimate_revenue
merged_revenue.columns = ['surf_revenue', 'ultimate_revenue']

# sort the index by the chronologically order of months
merged_revenue.index = pd.CategoricalIndex(merged_revenue.index, categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
merged_revenue.sort_index(inplace=True)

# Plot the bar plot
merged_revenue.plot(kind = 'bar', figsize = (15, 10))
plt.title('Average revenue per month for each plan')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()


# In[43]:


# Compare the amount of messages sent by users per plan

# Get the data for the surf plan and plot the histogram
surf_data = merged_data[merged_data['plan_name'] == 'surf']['revenue'].hist(bins=50, figsize = (15, 10), label = 'surf')

# Get the data for the ultimate plan and plot the histogram

ultimate_data = merged_data[merged_data['plan_name'] == 'ultimate']['revenue'].hist(bins=50, figsize = (15, 10), label = 'ultimate')

# Set the title and labels
plt.title('Average Revenue Brought in for Both Plan')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[44]:


# Calculate the mean and the variance of the monthly revenue

# Get the data for the surf plan
surf_data_revenue = merged_data[merged_data['plan_name'] == 'surf'].groupby('user_id')['revenue'].mean()

# Get the data for the ultimate plan
ultimate_data_revenue = merged_data[merged_data['plan_name'] == 'ultimate'].groupby('user_id')['revenue'].mean()

# Calculate the mean, variance, and standard deviation for the surf plan
surf_mean_revenue = surf_data_revenue.mean()
surf_var_revenue = surf_data_revenue.var()
surf_std_revenue = surf_data_revenue.std()


# Calculate the mean, variance, and standard deviation for the ultimate plan
ultimate_mean_revenue = ultimate_data_revenue.mean()
ultimate_var_revenue = ultimate_data_revenue.var()
ultimate_std_revenue = ultimate_data_revenue.std()


# Print the results
print(f'Mean for the surf plan: {round(surf_mean_revenue,2)}')
print(f'Variance for the surf plan: {round(surf_var_revenue,2)}')
print(f'Standard deviation for the surf plan: {round(surf_std_revenue,2)}')
print(f'Mean for the ultimate plan: {round(ultimate_mean_revenue,2)}')
print(f'Variance for the ultimate plan: {round(ultimate_var_revenue,2)}')
print(f'Standard deviation for the ultimate plan: {round(ultimate_std_revenue,2)}')


# In[45]:


# Plot a boxplot to visualize the distribution of the monthly revenue
sns.boxplot(x = 'plan_name', y = 'revenue', data = merged_data, palette = 'Set2')
plt.title('Boxplot of Average Monthly Revenue')
plt.show()


# Looking at the above boxplot, the Ultimate plan brings in more revenue with its higher monthly fee and mainly stay at 70 USD. There are a good portion of clients that are in the Surf plan that pay more than the standard 20 USD monthly fee. The data on the Surf plan may indicate room for Megaline to capitalize on this to bring in more revenue.

#  

# # Test statistical hypotheses

# [Test the hypothesis that the average revenue from users of the Ultimate and Surf calling plans differs.]

# [Formulate the null and the alternative hypotheses, choose the statistical test, decide on the alpha value.]

# In[46]:


# Test the hypotheses

# Obtain the desired data slices
surf_data = merged_data[merged_data['plan_name'] == 'surf'].groupby('user_id')['revenue'].mean()
ultimate_data = merged_data[merged_data['plan_name'] == 'ultimate'].groupby('user_id')['revenue'].mean()

# The alpha value, or statistical significance, will be 5%
alpha = 0.05

# Calculate the result using the t-test
# equal_var will be set to false since we know that the variance is not equal
results = st.ttest_ind(surf_data, ultimate_data, equal_var=False)

# Print the results
print('p-value', round(results.pvalue,7))

# Determine if we can or cannot reject the null hypothesis and print the result
if results.pvalue < alpha:
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# Test the hypothesis that the average revenue from users in the NY-NJ area is different from that of the users from the other regions.Formulate the null and the alternative hypotheses, choose the statistical test, decide on the alpha value

# In[47]:


# Test the hypotheses

# Obtain the desired data slics
df_with_location = merged_data.merge(right=users, on='user_id')
df_NY_NJ = df_with_location.query('city == "New York-Newark-Jersey City, NY-NJ-PA MSA"')
df_other_regions = df_with_location.query('city != "New York-Newark-Jersey City, NY-NJ-PA MSA"')

# Calculate the average monthly revenue collected from each user in each data slice
df_NY_NJ = df_NY_NJ.groupby('user_id')['revenue'].mean()
df_other_regions = df_other_regions.groupby('user_id')['revenue'].mean()

# Set the alpha value to 0.05
alpha = 0.05

# Run the t-test, pass equal_var=True
results = st.ttest_ind(df_NY_NJ, df_other_regions, equal_var=True)

# Print the p-value results
print('p-value', results.pvalue)

# Determine if we can or cannot reject the null hypothesis and print the result
if results.pvalue < alpha:
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# # General conclusion
# 
# The tests that were constructed pointed out the hypothesis that the average monthly revenue for clients plan were equal to each other. However, we were not able reject the hypothesis that the average monthly revenue for clintes that were in the NY-NJ-PA areas were different than the average monthly revenue for the clients for other regions. 
# Looking at the data and my own previous obervations, it looks like the Surf plan has a much greater number of clients and the average amount of monthly revenue that comes from the Surf plan is very significant and shows opportunity to increase revenue that can be brought in through increased monthly fees or penalty fees. In the Ultimate plan those clients mainly stayed within their internet usage and there are fewer clients in that plan versus the amount of clients in the Surf plan. Would advise focusing bringing clients into Surf plan more in order to produce more revenue in the future.

#  
