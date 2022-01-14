# python_repository
Python rep Reyn


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

