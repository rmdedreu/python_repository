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

