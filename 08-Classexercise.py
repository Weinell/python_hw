from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import minmax_scale as mm

## Part 1

df = pd.read_csv('../data/ds_salaries.csv')
print(df.head())

df = df[['work_year','experience_level','salary_in_usd','job_title','company_location','company_size']]
print(df.head())

df = pd.get_dummies(df, columns=['experience_level'])
print(df.head())

df['company_location'] = le.fit_transform(df,df['company_location'])
print(df.head())
print(df['company_location'])


df['company_location'] = mm(df['company_location'], feature_range=(0,1))
print(df.head())
print(df['company_location'])


bin = [0,75000,120000,250000,600000]
label = ['S$','M$','L$','XL$']
df['binned_salary'] = pd.cut(df['salary_in_usd'], bins=bin, labels=label)
print(df.head())
print(df['binned_salary'])

quartiles = [0, .25, .5, .75, 1.]
df['binned_salary_qcut'] = pd.qcut(df['salary_in_usd'], q=4,labels=label)
print(df['binned_salary_qcut'])


df = df.groupby(['binned_salary_qcut','company_size']).size()
unstacked = df.unstack()
print(unstacked)
unstacked.plot.bar()
plt.show()


## Part 2
"""
def plot_CompanySize_or_Experience(df,input='company_size'):
    df = df.groupby(['binned_salary_qcut',{input}]).size()
    unstacked = df.unstack()
    print(unstacked)
    unstacked.plot.bar()
    plt.show()

plot_CompanySize_or_Experience(df)
"""



## Part 3

shoppers = {
    'Paula':{'Is':4,'Juice':2,'Kakao':3,'Lagkager':2},
    'Peter':{'Is':2,'Juice':5,'Kakao':0, 'Lagkager':4},
    'Pandora':{'Is':5,'Juice':3, 'Kakao':4, 'Lagkager':5},
    'Pietro':{'Is':1,'Juice':8, 'Kakao':9, 'Lagkager':1}
}
shop_prices = {
    'Netto': {'Is':10.50,'Juice':2.25,'Kakao':4.50,'Lagkager':33.50},
    'Fakta': {'Is':4.00,'Juice':4.50,'Kakao':6.25,'Lagkager':20.00}
}
print(shoppers)
shoppers_t = pd.DataFrame(shoppers).T
print(shoppers_t)
shop_prices_t = pd.DataFrame(shop_prices).T
print(shop_prices_t)

def minimizeShoppingNeeds():
    shopper_store = {}
    for shopper, item in shoppers_t.iterrows():
        print(shopper)
        store_price_times_shopper = pd.DataFrame(item*shop_prices_t)
        sum_store_price = store_price_times_shopper.sum(axis=1)
        print(sum_store_price)
        store_to_buy_from = min(sum_store_price.index)
        shopper_store.update({shopper : store_to_buy_from})
    return shopper_store

print(minimizeShoppingNeeds())
