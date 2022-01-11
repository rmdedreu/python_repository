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
