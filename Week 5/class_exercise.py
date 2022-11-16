import pandas as pd
import matplotlib.pyplot as plt

# 1 
df = pd.read_csv('../../data/titanic_train.csv')
print(df.head())

# 2
df = df.set_index('PassengerId')
print(df.head())

# 3
df = df.sort_values('Fare',ascending=True)
print(df.head())

testname = 'Weinell, Mr. Nikolaj'
# 4
def swapFirstLastName(name):
    first_name = name.split(', ')[1].split(' ')[1].strip()
    last_name = name.split(',')[0].strip()
    mergeName = f'{first_name} {last_name}'
    return mergeName

print(swapFirstLastName(testname))

# 5
df['Name'] = df['Name'].apply(swapFirstLastName)
print(df.head())

# 6
def onlyLastNameBeforeSwap(name):
    split = name.split(', ')
    return split[0]

def onlyLastNameAfterSwap(name):
    split = name.split(' ')
    return split[1]

df['Name'] = df['Name'].apply(onlyLastNameAfterSwap)
print(df)

# 7
df_tickets = df[df['Ticket'].isin(['350406','248706','382652','244373','345763','2649','239865'])]
print(df_tickets)

# 8
df_age = df[df['Age'].notna()]
print(df_age)

# 9
df_age["Age"] = pd.to_numeric(df_age["Age"])
print(df_age)

# 10
df_age = df_age[df_age['Age'] < 19]
print(df_age)

# 11
df = df[df['Cabin'].notna()]
print(df)

# 12


# 13
df_original = pd.read_csv('../../data/titanic_train.csv')
df_original['Relations'] = df_original['SibSp'] + df_original['Parch']
print(df_original)

# 14
df_original.plot(x='Relations', y='Survived', style='o')
plt.show()
