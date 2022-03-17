# python_repository
In this repository I store my python code


## using a lambda function to split a variable and add it to the dataframe

import codecademylib3
import pandas as pd

df = pd.read_csv('employees.csv')

get_last_name = lambda x: x.split()[-1]

df['last_name'] = df.name.apply(get_last_name)

print(df)

## Applying a Lambda to a Row

total_earned = lambda row: (row.hourly_wage * 40) + ((row.hourly_wage * 1.5) * (row.hours_worked - 40)) \
	if row.hours_worked > 40 \
  else row.hourly_wage * row.hours_worked
  
df['total_earned'] = df.apply(total_earned, axis = 1)

print(df)

## rename columns using df.rename 
(inplace makes sure you dont replace the dataframe)

df.rename(columns={
    'name': 'First Name',
    'age': 'Age'},
    inplace=True)
    
    
 ## Other examples of creating new columns using lambda
 
 import codecademylib3
import pandas as pd

orders = pd.read_csv('shoefly.csv')

print(orders.head(5))

orders['shoe_source'] = orders.shoe_material.apply(lambda x: \
                        	'animal' if x == 'leather'else 'vegan')

orders['salutation'] = orders.apply(lambda row: \
                                    'Dear Mr. ' + row['last_name']
                                    if row['gender'] == 'male'
                                    else 'Dear Ms. ' + row['last_name'],
                                    axis=1)

## even more examples for lambda and modifying dataframes

import codecademylib3
import pandas as pd 

inventory = pd.read_csv('inventory.csv')

print(inventory.head(11))

staten_island = inventory.head(11)

print(staten_island)

product_request = staten_island['product_description']

seed_request = inventory.loc[(inventory['location'] == "Brooklyn") & (inventory['product_type'] == "seeds")]

inventory['in_stock'] = inventory.quantity.apply(lambda x: True if x > 0 else False)



inventory['total_value'] = inventory.price * inventory.quantity

print(inventory)

combine_lambda = lambda row: \
    '{} - {}'.format(row.product_type,
                     row.product_description)

inventory['full_description'] = inventory.apply(combine_lambda, axis = 1)

print(inventory)

## Pivot a groupby

print(shoe_counts)
shoe_counts_pivot = shoe_counts.pivot(
    columns='shoe_color',
    index='shoe_type',
    values='id').reset_index()
    
 ## more groupby and pivot
 
user_visits = pd.read_csv('page_visits.csv')

print(user_visits.head())

click_source = user_visits.groupby("utm_source").id.count().reset_index()

print(click_source)

click_source_by_month = user_visits.groupby(["utm_source", "month"]).id.count().reset_index()

print(click_source_by_month)

click_source_by_month_pivot = click_source_by_month.pivot( columns='month', index='utm_source', values='id').reset_index()

print(click_source_by_month_pivot)

## check if null

ad_clicks['is_click'] = ~ad_clicks\
   .ad_click_timestamp.isnull()


## A/B test

import codecademylib3
import pandas as pd

ad_clicks = pd.read_csv('ad_clicks.csv')

print(ad_clicks.head())

click_source = ad_clicks.groupby("utm_source").user_id.count().reset_index()

print(click_source)

ad_clicks['is_click'] = ~ad_clicks\
   .ad_click_timestamp.isnull()

print(ad_clicks)

clicks_by_source = ad_clicks.groupby(["utm_source", "is_click"]).user_id.count().reset_index()

print(clicks_by_source)

clicks_pivot = clicks_by_source.pivot( columns='is_click', index='utm_source', values='user_id').reset_index()

print(clicks_pivot)

clicks_pivot['percent_click'] = \
   clicks_pivot[True] / \
   (clicks_pivot[True] + 
    clicks_pivot[False])

print(clicks_pivot)

ads = ad_clicks.groupby("experimental_group").user_id.count().reset_index()

print(ads)

percab = ad_clicks.groupby(["experimental_group", "is_click"]).user_id.count().reset_index()

pivot_percab = percab.pivot( columns='is_click', index='experimental_group', values='user_id').reset_index()

print(pivot_percab)

a_clicks = ad_clicks[
   ad_clicks.experimental_group
   == 'A']

b_clicks = ad_clicks[
   ad_clicks.experimental_group
   == 'B']

print(a_clicks) 

a_perc = a_clicks.groupby(["is_click", "day"]).user_id.count().reset_index()

aperc_pivot = a_perc.pivot( columns='is_click', index='day', values='user_id').reset_index()

aperc_pivot['percent_click'] = \
   aperc_pivot[True] / \
   (aperc_pivot[True] + 
    aperc_pivot[False])

print(aperc_pivot)

b_perc = b_clicks.groupby(["is_click", "day"]).user_id.count().reset_index()

bperc_pivot = b_perc.pivot( columns='is_click', index='day', values='user_id').reset_index()

bperc_pivot['percent_click'] = \
   bperc_pivot[True] / \
   (bperc_pivot[True] + 
    bperc_pivot[False])

print(bperc_pivot)



## multiple tables in pandas

orders_products = pd.merge(orders, products.rename(columns={'id': 'product_id'}))

pd.merge(
    orders,
    customers,
    left_on='customer_id',
    right_on='id',
    suffixes=['_order', '_customer']
)

store_a_b_outer = pd.merge(store_a, store_b, how='outer')


## mulitple left joins

all_data = visits.merge(cart, how='left')\
                .merge(checkout, how='left')\
                .merge(purchase, how='left')


## colors and markers in matplotlib 

plt.plot(time, revenue, color='purple', linestyle='--')

plt.plot(time, costs, color='#82edc9', marker='s')

## subplots

# First Subplot
plt.subplot(1, 2, 1)
plt.plot(months, temperature, color='green')
plt.title('First Subplot')
 
# Second Subplot
plt.subplot(1, 2, 2)
plt.plot(months, flights_to_hawaii, "o", color='steelblue')
plt.title('Second Subplot')

#Top plot
plt.subplot(2, 1, 1)
plt.plot(straight_line,x)

#left bottom lplot
plt.subplot(2,2,3)
plt.plot(x, parabola)

#right bottom lplot
plt.subplot(2,2,4)
plt.plot(x, cubic)

plt.subplots_adjust(wspace=0.35)
plt.subplots_adjust(bottom=0.2)

plt.show

## A graph showing conversions over time using matplotlib

import codecademylib
from matplotlib import pyplot as plt

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep","Oct", "Nov", "Dec"]

months = range(12)
conversion = [0.05, 0.08, 0.18, 0.28, 0.4, 0.66, 0.74, 0.78, 0.8, 0.81, 0.85, 0.85]

plt.xlabel("Months")
plt.ylabel("Conversion")

plt.plot(months, conversion)

# Your work here
ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)
ax.set_yticks([0.1, 0.25, 0.5, 0.75])
ax.set_yticklabels(['10%', '25%', '50%', '75%'])


plt.show()


## another graph

import codecademylib
from matplotlib import pyplot as plt

x = range(12)
y1 = [0.05, 0.08, 0.18, 0.28, 0.4, 0.66, 0.74, 0.78, 0.8, 0.81, 0.85, 0.85]
y2 = [0.06, 0.10, 0.12, 0.24, 0.4, 0.5, 0.68, 0.70, 0.75, 0.81, 0.85, 0.85]

plt.plot(x, y1, color='pink', linestyle='--', marker='o')
plt.title('Two Lines on One Graph')
plt.plot(x, y2, color='gray', linestyle='-', marker='o')
plt.xlabel("Amazing X-axis") 
plt.ylabel("Incredible Y-axis")
plt.legend(["label1", "label2"], loc=4)

plt.show()

## more graphs

import codecademylib
from matplotlib import pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

visits_per_month = [9695, 7909, 10831, 12942, 12495, 16794, 14161, 12762, 12777, 12439, 10309, 8724]

# numbers of limes of different species sold each month
key_limes_per_month = [92.0, 109.0, 124.0, 70.0, 101.0, 79.0, 106.0, 101.0, 103.0, 90.0, 102.0, 106.0]
persian_limes_per_month = [67.0, 51.0, 57.0, 54.0, 83.0, 90.0, 52.0, 63.0, 51.0, 44.0, 64.0, 78.0]
blood_limes_per_month = [75.0, 75.0, 76.0, 71.0, 74.0, 77.0, 69.0, 80.0, 63.0, 69.0, 73.0, 82.0]


# create your figure here
plt.figure(figsize = (12, 8))
ax1 = plt.subplot(1, 2, 1)
x_values = range(len(months))
plt.plot(x_values, visits_per_month, marker = 'o')
plt.xlabel('Months')
plt.ylabel('Visits Per Month')
ax1.set_xticks(x_values)
ax1.set_xticklabels(months)
plt.title('Visitors Per Month')
plt.subplots_adjust(hspace = 0.55)

ax2 = plt.subplot(1, 2, 2)
plt.plot(x_values, key_limes_per_month, color = 'green', marker = 's')
plt.plot(x_values, persian_limes_per_month, color = 'blue', marker = '*')
plt.xlabel('Months')
plt.ylabel('Number of Limes Sold')
plt.plot(x_values, blood_limes_per_month, color = 'brown', marker = 'o')
plt.legend(['key limes', 'persian limes', 'blood limes'])
ax2.set_xticks(x_values)
ax2.set_xticklabels(months)
plt.title('Limes Sold Per Month')
plt.subplots_adjust(hspace = 0.55)
plt.savefig('Customers and Their Sales Habits.png')

plt.show()

## Side-By-Side bars

import codecademylib
from matplotlib import pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]

#Paste the x_values code here
n = 1  # This is our first dataset (out of 2)
t = 2 # Number of datasets
d = 6 # Number of sets of bars
w = 0.8 # Width of each bar
store1_x = [t*element + w*n for element
             in range(d)]

plt.bar(store1_x, sales1)       

n = 2  # This is our second dataset (out of 2)
t = 2 # Number of datasets
d = 6 # Number of sets of bars
w = 0.8 # Width of each bar
store2_x = [t*element + w*n for element
             in range(d)]

plt.bar(store2_x, sales2)
plt.show()

## stacked bars

import codecademylib
from matplotlib import pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]
  
plt.bar(range(len(drinks)), sales1)
plt.bar(range(len(drinks)), sales2, bottom=sales1)

plt.legend(["Location 1", "Location 2"])

plt.show()

## error bars

import codecademylib
from matplotlib import pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
ounces_of_milk = [6, 9, 4, 0, 9, 0]
error = [0.6, 0.9, 0.4, 0, 0.9, 0]

# Plot the bar graph here
plt.bar(range(len(drinks)), ounces_of_milk, yerr=error, capsize=5)

plt.show()

## fill between (error margin collored) 

import codecademylib
from matplotlib import pyplot as plt

months = range(12)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
revenue = [16000, 14000, 17500, 19500, 21500, 21500, 22000, 23000, 20000, 19500, 18000, 16500]

plt.plot(months, revenue)

ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)

y_upper = [i + (i*0.10) for i in revenue]
y_lower = [i - (i*0.10) for i in revenue]

plt.fill_between(months, y_lower, y_upper, alpha=0.2)

plt.show()


## pie chart

import codecademylib
from matplotlib import pyplot as plt
import numpy as np

payment_method_names = ["Card Swipe", "Cash", "Apple Pay", "Other"]
payment_method_freqs = [270, 77, 32, 11]

#make your pie chart here
plt.pie(payment_method_freqs)
plt.axis('equal')

plt.show()


## pie chart labeling

import codecademylib
from matplotlib import pyplot as plt

payment_method_names = ["Card Swipe", "Cash", "Apple Pay", "Other"]
payment_method_freqs = [270, 77, 32, 11]

plt.pie(payment_method_freqs, autopct="%0.1f%%")
plt.axis('equal')
plt.legend(payment_method_names)

plt.show()

## multiple histograms

import codecademylib
from matplotlib import pyplot as plt
from script import sales_times1
from script import sales_times2

plt.hist(sales_times1, bins=20, alpha=0.4, normed=True)
#plot your other histogram here
plt.hist(sales_times2, bins=20, alpha=0.4, normed=True)

plt.show()

## final exam averages

import codecademylib
from matplotlib import pyplot as plt

past_years_averages = [82, 84, 83, 86, 74, 84, 90]
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006]
error = [1.5, 2.1, 1.2, 3.2, 2.3, 1.7, 2.4]

# Make your chart here
plt.figure(figsize=(10,8))
plt.bar(range(len(past_years_averages)), past_years_averages, yerr = error, capsize=5)
plt.axis([-0.5, 6.5, 70, 95])

ax = plt.subplot()
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)

plt.title('Final Exam Averages')
plt.ylabel('Test average')
plt.xlabel('Year')

plt.savefig('my_bar_chart.png')

plt.show()

## more stacked bars

import codecademylib
from matplotlib import pyplot as plt
import numpy as np

unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']
As = [6, 3, 4, 3, 5]
Bs = [8, 12, 8, 9, 10]
Cs = [13, 12, 15, 13, 14]
Ds = [2, 3, 3, 2, 1]
Fs = [1, 0, 0, 3, 0]

x = range(5)

c_bottom = np.add(As, Bs)
d_bottom = np.add(c_bottom, Cs)
f_bottom = np.add(d_bottom, Ds)
#create d_bottom and f_bottom here

#create your plot here
plt.figure(figsize=(10,8))
plt.bar(x, As)
plt.bar(x, Bs, bottom=As)
plt.bar(x, Cs, bottom=c_bottom)
plt.bar(x, Ds, bottom=d_bottom)
plt.bar(x, Fs, bottom=f_bottom)

ax = plt.subplot()
ax.set_xticks(range(len(unit_topics)))
ax.set_xticklabels(unit_topics)

plt.title('Grade distribution')
plt.xlabel('Unit')
plt.ylabel('Number of Students')
plt.show()
plt.savefig('my_stacked_bar.png')

## two histograms in a plot

import codecademylib
from matplotlib import pyplot as plt

exam_scores1 = [62.58, 67.63, 81.37, 52.53, 62.98, 72.15, 59.05, 73.85, 97.24, 76.81, 89.34, 74.44, 68.52, 85.13, 90.75, 70.29, 75.62, 85.38, 77.82, 98.31, 79.08, 61.72, 71.33, 80.77, 80.31, 78.16, 61.15, 64.99, 72.67, 78.94]
exam_scores2 = [72.38, 71.28, 79.24, 83.86, 84.42, 79.38, 75.51, 76.63, 81.48,78.81,79.23,74.38,79.27,81.07,75.42,90.35,82.93,86.74,81.33,95.1,86.57,83.66,85.58,81.87,92.14,72.15,91.64,74.21,89.04,76.54,81.9,96.5,80.05,74.77,72.26,73.23,92.6,66.22,70.09,77.2]

# Make your plot here
plt.figure(figsize=(10, 8))
plt.hist(exam_scores1, bins=12, normed=True, histtype='step', linewidth=2)
plt.hist(exam_scores2, bins=12, normed=True, histtype='step', linewidth=2)

plt.legend(["1st Yr Teaching", "2nd Yr Teaching"])

plt.title("Final Exam Score Distribution")
plt.xlabel("Percentage")
plt.ylabel("Frequency")

plt.show()
plt.savefig('my_histogram.png')

## labeled pie chart

import codecademylib
from matplotlib import pyplot as plt

unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']
num_hardest_reported = [1, 3, 10, 15, 1]

#Make your plot here
plt.figure(figsize=(10,8))
plt.pie(num_hardest_reported, labels=unit_topics, autopct="%1d%%")

plt.axis('equal')
plt.title('Hardest Topics')

plt.show()
plt.savefig("my_pie_chart.png")

## line with shaded error

import codecademylib
from matplotlib import pyplot as plt

hours_reported =[3, 2.5, 2.75, 2.5, 2.75, 3.0, 3.5, 3.25, 3.25,  3.5, 3.5, 3.75, 3.75,4, 4.0, 3.75,  4.0, 4.25, 4.25, 4.5, 4.5, 5.0, 5.25, 5, 5.25, 5.5, 5.5, 5.75, 5.25, 4.75]
exam_scores = [52.53, 59.05, 61.15, 61.72, 62.58, 62.98, 64.99, 67.63, 68.52, 70.29, 71.33, 72.15, 72.67, 73.85, 74.44, 75.62, 76.81, 77.82, 78.16, 78.94, 79.08, 80.31, 80.77, 81.37, 85.13, 85.38, 89.34, 90.75, 97.24, 98.31]

# Create your figure here
plt.figure(figsize=(10,8))

# Create your hours_lower_bound and hours_upper_bound lists here 
hours_lower_bound = [element - (element * 0.2) for element in hours_reported]
hours_upper_bound = [element + (element * 0.2) for element in hours_reported]

# Make your graph here
plt.plot(exam_scores, hours_reported, linewidth=2)

plt.fill_between(exam_scores, hours_lower_bound, hours_upper_bound, alpha = 0.2)

plt.xlabel("Score")
plt.title("Time spent studying vs final exam scores")
plt.ylabel('Hours studying (self-reported)')
plt.show()
plt.savefig("my_line_graph.png")


## std lines , personal line, in bar chart

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from data import nba_data, okcupid_data

nba_mean = np.mean(nba_data)
okcupid_mean = np.mean(okcupid_data)

#Change this variable to your height (in inches)!
your_height = 0

nba_standard_deviation = np.std(nba_data)
okcupid_standard_deviation = np.std(okcupid_data)

plt.subplot(211)
plt.title("NBA Player Heights")
plt.xlabel("Height (inches)")

plt.hist(nba_data)

plt.axvline(nba_mean, color='#FD4E40', linestyle='solid', linewidth=2, label = "Mean")

plt.axvline(nba_mean + nba_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2, label = "Standard Deviations")
plt.axvline(nba_mean - nba_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(nba_mean + nba_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(nba_mean - nba_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(nba_mean + nba_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(nba_mean - nba_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(your_height, color='#62EDBF', linestyle='solid', linewidth=2, label = "You")

plt.xlim(55, 90)
plt.legend()


plt.subplot(212)
plt.title("OkCupid Profile Heights")
plt.xlabel("Height (inches)")

plt.hist(okcupid_data)

plt.axvline(okcupid_mean, color='#FD4E40', linestyle='solid', linewidth=2, label = "Mean")

plt.axvline(okcupid_mean + okcupid_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2, label = "Standard Deviations")
plt.axvline(okcupid_mean - okcupid_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(okcupid_mean + okcupid_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(okcupid_mean - okcupid_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(okcupid_mean + okcupid_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(okcupid_mean - okcupid_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(your_height, color='#62EDBF', linestyle='solid', linewidth=2, label = "You")

plt.xlim(55, 90)
plt.legend()




plt.tight_layout()
plt.show()

## mean and std for every month

for i in range(1, 13):
  month = london_data.loc[london_data["month"] == i]["TemperatureC"]
  print("The mean temperature in month "+str(i) +" is "+ str(np.mean(month)))
  print("The standard deviation of temperature in month "+str(i) +" is "+ str(np.std(month)) +"\n")
  
  
 ## basic statistics
 
 A p-value of 0.05 means that if the null hypothesis is true, there is a 5% chance that an observed sample statistic could have occurred due to random sampling error. For example, in comparing two sample means, a p-value of 0.05 indicates there is a 5% chance that the observed difference in sample means occurred by random chance, even though the population means are equal.
 
 Generally, we want a p-value of less than 0.05, meaning that there is less than a 5% chance that our results are due to random chance.
 
 ## Check p values of multiple colums 
 
for i in range(1000): # 1000 experiments
   tstatistic, pval = ttest_1samp(daily_prices[i], 1000)
   if pval < 0.05:
     incorrect_results += 1
     
  ## tukeys range test
  
  from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

stat, pval = f_oneway(a, b, c)
print pval

Using our data from ANOVA, we create v and l
v = np.concatenate([a, b, c])
labels = ['a'] * len(a) + ['b'] * len(b) + ['c'] * len(c)

tukey_results = pairwise_tukeyhsd(v, labels, 0.05)

print(tukey_results)

#binomial test

from scipy.stats import binom_test

pval = binom_test(510, n=10000, p=0.06)
print pval

pval2 = binom_test(590, n=10000, p=0.06)
print pval2

# sample size of a survey

import math

margin_of_error = 4
confidence_level = 95
likely_proportion = 40
population_size = 100000

sample_size = 573

//sample_size/120, rounded up:
//weeks_of_survey = 5
weeks_of_survey = math.ceil(sample_size/120.0)


# pie chart from restaurant types

import codecademylib3
from matplotlib import pyplot as plt
import pandas as pd

restaurants = pd.read_csv('restaurants.csv')

cuisine_counts = restaurants.groupby('cuisine')\
                            .name.count()\
                            .reset_index()

cuisines = cuisine_counts.cuisine.values
counts = cuisine_counts.name.values


plt.pie(counts,
        labels=cuisines,
       autopct='%d%%')
plt.title('FoodWheel')
plt.axis('equal')
plt.show()

# average orders over time with error bars

import codecademylib
from matplotlib import pyplot as plt
import pandas as pd

orders = pd.read_csv('orders.csv')

orders['month'] = orders.date.apply(lambda x: x.split('-')[0])

avg_order = orders.groupby('month').price.mean().reset_index()

std_order = orders.groupby('month').price.std().reset_index()

ax = plt.subplot()

bar_heights = avg_order.price
bar_errors = std_order.price

plt.bar(range(len(bar_heights)),
  			bar_heights,
        yerr=bar_errors,
       capsize=5)
ax.set_xticks(range(len(bar_heights)))
ax.set_xticklabels(['April', 'May', 'June', 'July', 'August', 'September'])
plt.ylabel('Average Order Amount')
plt.title('Order Amount over Time')
plt.show()

## customer expenditure over 6 months

import codecademylib
from matplotlib import pyplot as plt
import pandas as pd

orders = pd.read_csv('orders.csv')

customer_amount = orders.groupby('customer_id').price.sum().reset_index()

print(customer_amount.head())

plt.hist(customer_amount.price.values,
        range=(0, 200), bins=40)
plt.xlabel('Total Spent')
plt.ylabel("Number of Customers")
plt.title('Customer Expenditure Over 6 Months')

plt.show()

# purchase if not null in cell

import codecademylib
import pandas as pd

df = pd.read_csv('clicks.csv')

df['is_purchase'] = df.click_day.apply(
  lambda x: 'Purchase' if pd.notnull(x) else 'No Purchase'
)

purchase_counts = df.groupby(['group', 'is_purchase'])\
	.user_id.count().reset_index()

print purchase_counts
