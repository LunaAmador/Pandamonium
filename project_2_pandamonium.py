# Import 3rd party libraries
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

# Configure Notebook
import warnings
warnings.filterwarnings('ignore')

'''Extracting data from goodcarbadcar HTML tables for 2019, 2020, and 2021 and merging it into one dataframe'''

from requests import get

#2019 data
response = get('https://www.goodcarbadcar.net/2019-canada-vehicle-sales-figures-by-model/')
html_soup = BeautifulSoup(response.text, 'html.parser')
table_rows_2019 = html_soup.find_all('tr')[594:-1] #excludes the sum row

model_names = []
january = []
february = []
march = []
april = []
may = []
june = []
july = []
august = []
september = []
october = []
november = []
december = []

# iterate through each row in the table
for row in table_rows_2019:
    # use certain tags positions and/or subtags, then extract the text, and use .strip() to remove leading whitespace, and to list
    model_name = row.find_all('td')[0].text.strip()
    model_names.append(model_name)

    jan_sum = row.find_all('td')[1].text.strip()
    january.append(int(jan_sum.replace(',','')))

    feb_sum = row.find_all('td')[2].text.strip()
    february.append(int(feb_sum.replace(',','')))

    mar_sum = row.find_all('td')[3].text.strip()
    march.append(int(mar_sum.replace(',','')))

    apr_sum = row.find_all('td')[4].text.strip()
    april.append(int(apr_sum.replace(',','')))

    may_sum = row.find_all('td')[5].text.strip()
    may.append(int(may_sum.replace(',','')))

    jun_sum = row.find_all('td')[6].text.strip()
    june.append(int(jun_sum.replace(',','')))

    jul_sum = row.find_all('td')[7].text.strip()
    july.append(int(jul_sum.replace(',','')))

    aug_sum = row.find_all('td')[8].text.strip()
    august.append(int(aug_sum.replace(',','')))

    sep_sum = row.find_all('td')[9].text.strip()
    september.append(int(sep_sum.replace(',','')))

    oct_sum = row.find_all('td')[10].text.strip()
    october.append(int(oct_sum.replace(',','')))

    nov_sum = row.find_all('td')[11].text.strip()
    november.append(int(nov_sum.replace(',','')))

    dec_sum = row.find_all('td')[12].text.strip()
    december.append(int(dec_sum.replace(',','')))

# make dataframe containing with columns corresponding to the lists
car_sales_2019 = pd.DataFrame({'make_and_model': model_names, '2019-01-01': january,
                          '2019-02-01': february, '2019-03-01': march,'2019-04-01': april,
                          '2019-05-01':may, '2019-06-01':june, '2019-07-01':july,'2019-08-01':august,
                          '2019-09-01': september, '2019-10-01': october,
                          '2019-11-01': november,'2019-12-01': december}).replace("-"," ")

#monthly_sums_2019 = list(car_sales_2019.sum(axis = 0)[1:])

#print(monthly_sums_2019)

##################################################################################
#2020 data
response = get('https://www.goodcarbadcar.net/2020-canada-vehicle-sales-figures-by-model/')
html_soup = BeautifulSoup(response.text, 'html.parser')
table_rows_2020 = html_soup.find_all('tr')[1:-1] #excludes the sum row

model_names = []
january = []
february = []
march = []
april = []
may = []
june = []
july = []
august = []
september = []
october = []
november = []
december = []

# iterate through each row in the table
for row in table_rows_2020:
    # use certain tags positions and/or subtags, then extract the text, and use .strip() to remove leading whitespace, and to list
    model_name = row.find_all('td')[0].text.strip()
    model_names.append(model_name)

    jan_sum = row.find_all('td')[1].text.strip()
    january.append(int(jan_sum.replace(',','')))

    feb_sum = row.find_all('td')[2].text.strip()
    february.append(int(feb_sum.replace(',','')))

    mar_sum = row.find_all('td')[3].text.strip()
    march.append(int(mar_sum.replace(',','')))

    apr_sum = row.find_all('td')[4].text.strip()
    april.append(int(apr_sum.replace(',','')))

    may_sum = row.find_all('td')[5].text.strip()
    may.append(int(may_sum.replace(',','')))

    jun_sum = row.find_all('td')[6].text.strip()
    june.append(int(jun_sum.replace(',','')))

    jul_sum = row.find_all('td')[7].text.strip()
    july.append(int(jul_sum.replace(',','')))

    aug_sum = row.find_all('td')[8].text.strip()
    august.append(int(aug_sum.replace(',','')))

    sep_sum = row.find_all('td')[9].text.strip()
    september.append(int(sep_sum.replace(',','')))

    oct_sum = row.find_all('td')[10].text.strip()
    october.append(int(oct_sum.replace(',','')))

    nov_sum = row.find_all('td')[11].text.strip()
    november.append(int(nov_sum.replace(',','')))

    dec_sum = row.find_all('td')[12].text.strip()
    december.append(int(dec_sum.replace(',','')))

# make dataframe containing with columns corresponding to the lists
car_sales_2020 = pd.DataFrame({'make_and_model': model_names,'2020-01-01': january,
                          '2020-02-01': february, '2020-03-01': march,'2020-04-01': april,
                          '2020-05-01':may, '2020-06-01':june, '2020-07-01':july,'2020-08-01':august,
                          '2020-09-01': september, '2020-10-01': october,
                          '2020-11-01': november,'2020-12-01': december}).replace("-"," ")

#monthly_sums_2020 = list(car_sales_2020.sum(axis = 0)[1:])

#print(monthly_sums_2020)

##################################################################################
#2021 data
response = get('https://www.goodcarbadcar.net/2021-canada-vehicle-sales-figures-by-model/')
html_soup = BeautifulSoup(response.text, 'html.parser')
table_rows_2021 = html_soup.find_all('tr')[580:-1] #excludes the sum row

model_names = []
january = []
february = []
march = []
april = []
may = []
june = []
july = []
august = []
september = []
october = []
november = []
december = []

# iterate through each row in the table
for row in table_rows_2021:
    # use certain tags positions and/or subtags, then extract the text, and use .strip() to remove leading whitespace, and to list
    model_name = row.find_all('td')[0].text.strip()
    model_names.append(model_name)

    jan_sum = row.find_all('td')[1].text.strip()
    january.append(int(jan_sum.replace(',','')))

    feb_sum = row.find_all('td')[2].text.strip()
    february.append(int(feb_sum.replace(',','')))

    mar_sum = row.find_all('td')[3].text.strip()
    march.append(int(mar_sum.replace(',','')))

    apr_sum = row.find_all('td')[4].text.strip()
    april.append(int(apr_sum.replace(',','')))

    may_sum = row.find_all('td')[5].text.strip()
    may.append(int(may_sum.replace(',','')))

    jun_sum = row.find_all('td')[6].text.strip()
    june.append(int(jun_sum.replace(',','')))

    jul_sum = row.find_all('td')[7].text.strip()
    july.append(int(jul_sum.replace(',','')))

    aug_sum = row.find_all('td')[8].text.strip()
    august.append(int(aug_sum.replace(',','')))

    sep_sum = row.find_all('td')[9].text.strip()
    september.append(int(sep_sum.replace(',','')))

    oct_sum = row.find_all('td')[10].text.strip()
    october.append(int(oct_sum.replace(',','')))

    nov_sum = row.find_all('td')[11].text.strip()
    november.append(int(nov_sum.replace(',','')))

    dec_sum = row.find_all('td')[12].text.strip()
    december.append(int(dec_sum.replace(',','')))

# make dataframe containing with columns corresponding to the lists
car_sales_2021 = pd.DataFrame({'make_and_model': model_names, '2021-01-01': january,
                          '2021-02-01': february, '2021-03-01': march,'2021-04-01': april,
                          '2021-05-01':may, '2021-06-01':june, '2021-07-01':july,'2021-08-01':august,
                          '2021-09-01': september, '2021-10-01': october,
                          '2021-11-01': november,'2021-12-01': december})

#monthly_sums_2021 = list(car_sales_2021.sum(axis = 0)[1:])

#print(monthly_sums_2021)

##################################################################################

#Merging 2019, 2020, and 2021 car sale data into one dataframe
car_sales = car_sales_2019.merge(right = car_sales_2020,
                                 how = 'outer',
                                 on = 'make_and_model').merge(right = car_sales_2021,
                                                          how = 'outer',
                                                          on = 'make_and_model')

##################################################################################
'''This section of the code will merge the car_sales data with the car_models data, such that every car make and model
sold will be placed in a car body category'''
#Car model csv file
car_models = pd.read_csv("Car_Model_List.csv")
missing_car_models = pd.read_csv("missing_models.csv")

months_for_analysis = ['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
                       '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01',
                       '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01',
                       '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',
                       '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
                       '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']

def make_and_model_canonicalization(car_sales, car_models,missing_car_models):
    '''Canonicalizes car_sales such that the elements in the make_and_model column of both dataframes matches
        input:
        car_sales --> A dataframe containing monthly car sales for different makes and models
        car_models --> A dataframe containing different car makes and models and their body

        returns:
        car_models dataset with modified strings in the make_and_model column to match the car_sales column canonicalization
        car_sales dataset with modified strings in the make_and_model column to match the car_models column canonicalization
        '''
    car_make_list = car_models["Make"].astype(str).str.lower().str.replace("-benz","").unique().tolist()
    # Creating a single column for make and model, to match the format of the car_sales column
    car_models["make_and_model"] = car_models["Make"].astype(str).str.lower() + " " + car_models["Model"].astype(str).str.lower().replace(car_make_list," ",regex=True)

    # Removing duplicates since the same make and model is present for multiple years in the car_models dataframe
    car_models = car_models.loc[:, ["make_and_model", "Category"]].drop_duplicates().apply(lambda x: x.replace("-"," ",regex=True)
                                                                                           .replace("/"," ",regex=True)
                                                                                           .replace("benz"," ",regex=True)
                                                                                           .replace("bolt ev","bolt",regex=True)
                                                                                           .replace("passenger"," ",regex=True)
                                                                                           .replace("crew cab"," ",regex=True)
                                                                                           .replace("extended cab"," ",regex=True)
                                                                                           .replace("regular cab"," ",regex=True)
                                                                                           .replace("1500 double cab"," ",regex=True)
                                                                                           .replace("2500 hd double cab"," ",regex=True)
                                                                                           .replace("2500 cargo"," ",regex=True)
                                                                                           .replace("2500 hd"," ",regex=True)
                                                                                           .replace("3500 hd"," ",regex=True)
                                                                                           .replace("1500"," ",regex=True)
                                                                                           .replace("fuel cell","fcv",regex=True)
                                                                                           .replace("electric"," ",regex=True)
                                                                                           .replace("boxster"," ",regex=True)
                                                                                           .replace("defender 90","defender",regex=True)
                                                                                           .replace("slc class","slc",regex=True))
    car_models["make_and_model"] = car_models["make_and_model"] \
        .apply(lambda x: x[:12] if ("ford transit" in x and "ford transit connect" not in x) else x)\
        .apply(lambda x: "ford f series" if "ford f150" in x else x)\
        .apply(lambda x: "ford e series" if "ford e350" in x else x)

    car_sales["make_and_model"] = car_sales["make_and_model"].str.lower()
    car_models["Category"] = car_models["Category"].replace("1992","",regex=True).replace("2020","",regex=True)

    car_sales = car_sales.apply(lambda x: x.replace("-", " ",regex=True)
                                .replace("/", " ",regex=True)
                                .replace("lr4"," ",regex=True)
                                .replace("impreza wrx","wrx",regex=True)
                                .replace("fr s"," ",regex=True)
                                .replace("benz"," ",regex=True)
                                .replace("etron","e tron",regex=True)
                                .replace("tuscon","tucson",regex=True)
                                .replace("mazda3","mazda 3",regex=True)
                                .replace("mazda6","mazda 6",regex=True)
                                .replace("nautilus"," ",regex=True)
                                .replace("90 series","s90",regex=True)
                                .replace("60 series","s60",regex=True)
                                .replace("40 series","s40",regex=True)
                                .replace("pickup"," ",regex=True)
                                .replace("family"," ",regex=True)
                                .replace("glk class"," ",regex=True)
                                .replace("gl gls class","gls",regex=True)
                                .replace("gle class","gle",regex=True)
                                .replace("slc class","slc",regex=True)
                                .replace("e   cls class","cls",regex=True))

    #Removing additional spaces in the strings in the male_and_model column
    car_sales["make_and_model"] = car_sales["make_and_model"].apply(lambda x:' '.join(x.split()))
    car_models["make_and_model"] = car_models["make_and_model"].apply(lambda x:' '.join(x.split()))
    car_models = pd.concat([car_models, missing_car_models], axis=0)

    return car_sales,car_models

#Merging car_sales and car_models by make and model
car_sales, car_models = make_and_model_canonicalization(car_sales, car_models,missing_car_models)
car_sales_by_size = car_sales.merge(right=car_models,
                                    how='outer',
                                    on='make_and_model')

#Filters out the vehicle makes and models that were not sold in any month from 2019-2021 in Canada
car_sales_by_size = car_sales_by_size[~car_sales_by_size[months_for_analysis].isna().all(1) | (car_sales_by_size[months_for_analysis]==0).all(1)]

#Drop duplicate rows
car_sales_by_size = car_sales_by_size.drop_duplicates(['make_and_model'])

#DEBUGGING: Finding null category values
#car_sales_list = car_sales["make_and_model"].to_list()
#missing_cars = (car_sales_by_size[car_sales_by_size["Category"].isnull()])["make_and_model"].to_list()
#print("Car bodies with null values: \n", missing_cars)
#print("Number of car bodies with null values: ", len(missing_cars))
#print("Car models list:\n", car_models["make_and_model"].to_list())


#put car model, car sales, and merged car sales list  to csv
car_models.to_csv('Car_Model_List_updated.csv', encoding='utf-8', index=False)
car_sales.to_csv('Car_Sales_2019_2021.csv', encoding='utf-8', index=False)
car_sales_by_size.to_csv('Merged Car Sales List.csv', encoding='utf-8', index=False)

#print(car_sales_by_size['Category'].unique())

#####################################################################################################################
#Now combining all the data into one table

#categorizing all combinations of car size category
small_cars=["Coupe","Hatchback","Convertible",'Convertible, Sedan','Coupe, Sedan, Convertible',
            'Coupe, Convertible','Convertible, Sedan, Coupe','Sedan, Hatchback','Hatchback, Sedan',
            'Hatchback, Sedan, Coupe','Convertible, Coupe, Hatchback']
midsize_cars=["SUV","Sedan","Wagon",'Wagon, Sedan']
large_cars=["Pickup","Van","Minivan",'Van Minivan']

#categorizing the sizes
car_sales_by_size['Category'] = car_sales_by_size['Category']\
    .apply(lambda x: 'small' if (x in small_cars) else x)\
    .apply(lambda x: 'midsize' if (x in midsize_cars) else x)\
    .apply(lambda x: 'large' if (x in large_cars) else x)

#create a new dataframe with categorized % columns
car_size_category = car_sales_by_size.groupby('Category').agg(sum)
#print(car_size_category)

#create monthly sums list
monthly_sums = list(car_sales_by_size.sum(axis = 0)[1:-1])

#calculate % of each size per month
car_size_category = (car_size_category/monthly_sums).T.rename(columns={"small": "prop_small",
                                                                       "midsize": "prop_midsize",
                                                                       "large": "prop_large"})

##sanity check
#car_size_category['total'] = list(car_size_category.sum(axis = 1))
#print(car_size_category)

#convert to datetime index full month name and full year
#car_size_category.index = car_size_category.reset_index()['index'].apply(lambda x: datetime.strptime(x,"%B %Y"))
car_size_category.index = pd.to_datetime(car_size_category.reset_index()['index'])

#create a column for monthly sums
car_size_category['total_sum'] = monthly_sums

car_size_category.to_csv('%_category.csv', encoding='utf-8', index=False)

#monthly employment stats units correct and datetime index set
employment_stats = pd.read_csv("monthly_employment_stats_2019-2021.csv").set_index('Labour force characteristics')\
    .T.astype(str).replace(',','',regex = True).astype(float)
employment_stats.index = employment_stats.reset_index()['index'].apply(lambda x: datetime.strptime(x,"%b-%y"))

employment_stats[['Unemployment rate','Participation rate','Employment rate']]= employment_stats[['Unemployment rate','Participation rate','Employment rate']]/100
employment_stats[employment_stats.columns.difference(['Unemployment rate','Participation rate','Employment rate'])]=employment_stats[employment_stats.columns.difference(['Unemployment rate','Participation rate','Employment rate'])]*1000

#employment_stats.to_csv('monthly_employment_stats_2019-2021_updated.csv', encoding='utf-8', index=False)

#monthly oil prices units correct and datetime index set
oil_prices = pd.read_csv("monthly_oil_prices_2019-2021.csv").rename(columns={"VALUE": "avg_oil_price"})\
                 .set_index('REF_DATE').loc[:,'avg_oil_price'].astype(float)*1000000
oil_prices.index = oil_prices.reset_index()['REF_DATE'].apply(lambda x: datetime.strptime(x,"%b-%y"))

#oil_prices.to_csv('monthly_oil_prices_2019-2021_updated.csv', encoding='utf-8', index=False)

#monthly transit ridership units correct and datetime index set
transit_ridership = pd.read_csv("monthly_transit_ridership_2019-2021.csv").rename(columns={"VALUE": "transit_ridership"})\
                        .set_index('REF_DATE').loc[:,'transit_ridership'].astype(float)*1000000
transit_ridership.index = transit_ridership.reset_index()['REF_DATE'].apply(lambda x: datetime.strptime(x,"%b-%y"))

#transit_ridership.to_csv('monthly_transit_ridership_2019-2021_updated.csv', encoding='utf-8', index=False)

#monthly dollar exchange rate USD-CAD datetime index set
der_usd_cad = pd.read_csv("monthly_der_2019-2021.csv").rename(columns={"FXMUSDCAD": "dollar_ex_rate"})\
                        .set_index('date').loc[:,'dollar_ex_rate'].astype(float)
der_usd_cad.index = der_usd_cad.reset_index()['date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))
#print(der_usd_cad)

#merging all dataframes into using datetime indices
total_table = pd.concat([car_size_category,employment_stats,oil_prices,transit_ridership,der_usd_cad], axis=1)
total_table.to_csv('Merged Total Table.csv', encoding='utf-8', index=False)

#####################################################################################################################

#Limiting table to independent variables for analysis and excluding pandemic dates from the beginning of the pandemic
#in March 2020 until December 2020, when there was a 0% change in Canadian GDP and after which GDP stabilized.
#https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2020009-eng.htm
pandemic_dates = pd.to_datetime(["2020-03-01","2020-04-01","2020-05-01","2020-06-01","2020-07-01","2020-08-01"
                                    ,"2020-09-01","2020-10-01","2020-11-01","2020-12-01","2021-01-01","2021-02-01",
                                 "2021-03-01","2021-04-01","2021-05-01","2021-06-01","2021-07-01","2021-08-01"
                                    ,"2021-09-01","2021-10-01","2021-11-01","2021-12-01"])
table_for_data_analysis = total_table.drop(pandemic_dates).reset_index().loc[:,["index","prop_large","prop_midsize"
                                                                                   ,"prop_small","Population"
                                                                                   ,"Employment","Full-time employment"
                                                                                   ,"Part-time employment","avg_oil_price"
                                                                                   ,"transit_ridership","dollar_ex_rate"]]

#Scatter plot of the purchases of small, midsize, and large vehicles
ax = sns.scatterplot(table_for_data_analysis, x="index", y="prop_small", label="Small cars")
ax = sns.scatterplot(table_for_data_analysis, x="index", y="prop_midsize", label="Midsize cars")
ax = sns.scatterplot(table_for_data_analysis, x="index", y="prop_large", label="Large cars")
plt.title('Purchases of Vehicles in Canada by Size January 2019 - December 2021', fontsize = 18)
ax.tick_params(axis='x', rotation=90)

#ax.xaxis.set_tick_params(labelsize = 14)
ax.yaxis.set_tick_params(labelsize = 14)
ax.set_ylim(0,1)
ax.set_xlabel('Date', fontsize = 18)
ax.set_ylabel('Car Sizes Purchased (%)', fontsize = 18)
ax.legend()
plt.show()

#Jointplots of oil prices, employment, population, and transit ridership with regards to vehicle sizes purchased
ax = sns.jointplot(table_for_data_analysis[["avg_oil_price","prop_large"]], x="avg_oil_price", y="prop_large", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Employment","prop_large"]], x="Employment", y="prop_large", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Full-time employment","prop_large"]], x="Full-time employment", y="prop_large", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Part-time employment","prop_large"]], x="Part-time employment", y="prop_large", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["transit_ridership","prop_large"]], x="transit_ridership", kind="reg", y="prop_large")
ax = sns.jointplot(table_for_data_analysis[["dollar_ex_rate","prop_large"]], x="dollar_ex_rate", kind="reg", y="prop_large")

ax = sns.jointplot(table_for_data_analysis[["avg_oil_price","prop_midsize"]], x="avg_oil_price", y="prop_midsize", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Employment","prop_midsize"]], x="Employment", y="prop_midsize", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Full-time employment","prop_midsize"]], x="Full-time employment", y="prop_midsize", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Part-time employment","prop_midsize"]], x="Part-time employment", y="prop_midsize", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["transit_ridership","prop_midsize"]], x="transit_ridership", y="prop_midsize", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["dollar_ex_rate","prop_midsize"]], x="dollar_ex_rate", y="prop_midsize", kind="reg")

ax = sns.jointplot(table_for_data_analysis[["avg_oil_price","prop_small"]], x="avg_oil_price", y="prop_small", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Employment","prop_small"]], x="Employment", y="prop_small", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Full-time employment","prop_small"]], x="Full-time employment", y="prop_small", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["Part-time employment","prop_small"]], x="Part-time employment", y="prop_small", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["transit_ridership","prop_small"]], x="transit_ridership", y="prop_small", kind="reg")
ax = sns.jointplot(table_for_data_analysis[["dollar_ex_rate","prop_small"]], x="dollar_ex_rate", y="prop_small", kind="reg")

plt.show()

#Check for potential outliers
print(round(table_for_data_analysis.describe(),2))

#Checking for null values
print(table_for_data_analysis.isnull().sum())

#Checking correlation between variables
correlation=table_for_data_analysis.corr()
plt.figure(figsize=(18,12))
plt.title('Correlation Heatmap of Vehicle Sales in Canada Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()

#####################################################################################################################
#Since populations and transit ridership were highly correlated, these will not be included in our model

#Splitting data into train, test, and validate
train, test = train_test_split(table_for_data_analysis, test_size=0.30, random_state=0)
test, val = train_test_split(test, test_size=0.50, random_state=0)

#Verify that data was split correctly
print('Train {}%'.format(train.shape[0] / table_for_data_analysis.shape[0] * 100))
print('Val {}%'.format(val.shape[0] / table_for_data_analysis.shape[0] * 100))
print('Test {}%'.format(test.shape[0] / table_for_data_analysis.shape[0] * 100))

def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def process_data(data):
    """Process the data for the guided model, which will predict the proportion of small, midsize, and
    large vehicles sold in Canada.
        Input: data --> dataframe with x and y values
        Output: X --> explanatory variables
                y1 --> proportion of small vehicles purchased
                y2 --> proportion of midsize vehicles purchased
                y3 --> proportion of large vehicles purchased
    """
    # Transform Data, Select Features
    data = select_columns(data,
                          'avg_oil_price',
                          'Full-time employment',
                          'Part-time employment',
                          'dollar_ex_rate',
                          'Employment',
                          'transit_ridership',
                          'prop_small',
                          'prop_midsize',
                          'prop_large')

    # Return predictors and response variables separately
    X = data.drop(['prop_small','prop_midsize','prop_large'], axis=1)
    y_small = data.loc[:, 'prop_small']
    y_midsize = data.loc[:, 'prop_midsize']
    y_large = data.loc[:, 'prop_large']

    return X, y_small, y_midsize, y_large

#Separating data into training, validation, and test datasets
X_train, y_train_small, y_train_midsize, y_train_large = process_data(train)
X_val, y_val_small, y_val_midsize, y_val_large = process_data(val)
X_test, y_test_small, y_test_midsize, y_test_large = process_data(test)

#Scaling variables based on minimum and maximum values
cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns = cols)
X_test = pd.DataFrame(X_test, columns = cols)

#Training linear regression models for small, midsize, and large vehicle sales
linear_model_small = lm.LinearRegression(fit_intercept=True)
linear_model_midsize = lm.LinearRegression(fit_intercept=True)
linear_model_large = lm.LinearRegression(fit_intercept=True)

linear_model_small.fit(X_train, y_train_small)
linear_model_midsize.fit(X_train, y_train_midsize)
linear_model_large.fit(X_train, y_train_large)

y_fitted_small = linear_model_small.predict(X_train)
y_fitted_midsize = linear_model_midsize.predict(X_train)
y_fitted_large = linear_model_large.predict(X_train)

y_predicted_small = linear_model_small.predict(X_val)
y_predicted_midsize = linear_model_small.predict(X_val)
y_predicted_large = linear_model_small.predict(X_val)

#Calculating root mean squared error for the predictive model
def rmse(actual, predicted):
    """
    Calculates RMSE from actual and predicted values
    Input:  actual (1D array) --> vector of actual values
            predicted (1D array) --> vector of predicted/fitted values
    Output:
      The root-mean square error of the input data, as a float
    """

    return ((((actual - predicted) ** 2).sum()) / len(actual)) ** 0.5

#Calculating error for each model
training_error_small = rmse(y_train_small,y_fitted_small)
training_error_midsize = rmse(y_train_midsize,y_fitted_midsize)
training_error_large = rmse(y_train_large,y_fitted_large)

val_error_small = rmse(y_val_small,y_predicted_small)
val_error_midsize = rmse(y_val_midsize,y_predicted_midsize)
val_error_large = rmse(y_val_large,y_predicted_large)

# Printing calculated error values for the training and validation sets of the three models
print('Training RMSE for small cars model: ${}'.format(training_error_small))
print('Training RMSE for midsize cars model: ${}'.format(training_error_midsize))
print('Training RMSE for large cars model: ${}'.format(training_error_large))

print('Validation RMSE for small cars model: ${}'.format(val_error_small))
print('Validation RMSE for midsize cars model: ${}'.format(val_error_midsize))
print('Validation RMSE for large cars model: ${}'.format(val_error_large))

#Cross validation RMSE