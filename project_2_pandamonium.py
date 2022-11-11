# Import 3rd party libraries
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime

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
car_sales_2019 = pd.DataFrame({'make_and_model': model_names, 'January 2019': january,
                          'February 2019': february, 'March 2019': march,'April 2019': april,
                          'May 2019':may, 'June 2019':june, 'July 2019':july,'August 2019':august,
                          'September 2019': september, 'October 2019': october,
                          'November 2019': november,'December 2019': december}).replace("-"," ")

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
car_sales_2020 = pd.DataFrame({'make_and_model': model_names, 'January 2020': january,
                          'February 2020': february, 'March 2020': march,'April 2020': april,
                          'May 2020':may, 'June 2020':june, 'July 2020':july,'August 2020':august,
                          'September 2020': september, 'October 2020': october,
                          'November 2020': november,'December 2020': december}).replace("-"," ")

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
car_sales_2021 = pd.DataFrame({'make_and_model': model_names, 'January 2021': january,
                          'February 2021': february, 'March 2021': march,'April 2021': april,
                          'May 2021':may, 'June 2021':june, 'July 2021':july,'August 2021':august,
                          'September 2021': september, 'October 2021': october,
                          'November 2021': november,'December 2021': december})

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

months_for_analysis = ['January 2019', 'February 2019', 'March 2019', 'April 2019', 'May 2019', 'June 2019',
                       'July 2019', 'August 2019', 'September 2019', 'October 2019', 'November 2019', 'December 2019',
                       'January 2020', 'February 2020', 'March 2020', 'April 2020', 'May 2020', 'June 2020',
                       'July 2020', 'August 2020', 'September 2020', 'October 2020', 'November 2020', 'December 2020',
                       'January 2021', 'February 2021', 'March 2021', 'April 2021', 'May 2021', 'June 2021',
                       'July 2021', 'August 2021', 'September 2021', 'October 2021', 'November 2021', 'December 2021']

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
                                .replace("impreza wrx","wrx ",regex=True)
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
car_sales_list = car_sales["make_and_model"].to_list()
missing_cars = (car_sales_by_size[car_sales_by_size["Category"].isnull()])["make_and_model"].to_list()
print("Car bodies with null values: \n", missing_cars)
print("Number of car bodies with null values: ", len(missing_cars))
print("Car models list:\n", car_models["make_and_model"].to_list())

#put car model, car sales, and merged car sales list  to csv
car_models.to_csv('Car_Model_List_updated.csv', encoding='utf-8', index=False)
car_sales.to_csv('Car_Sales_2019_2021.csv', encoding='utf-8', index=False)
car_sales_by_size.to_csv('Merged Car Sales List.csv', encoding='utf-8', index=False)

#print(car_sales_by_size['Category'].unique())

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
car_size_category = (car_size_category/monthly_sums).T.rename(columns={"small": "%_small",
                                                                       "midsize": "%_midsize",
                                                                       "large": "%_large"})

##sanity check
#car_size_category['total'] = list(car_size_category.sum(axis = 1))
#print(car_size_category)

#convert to datetime index full month name and full year
car_size_category.index = car_size_category.reset_index()['index'].apply(lambda x: datetime.strptime(x,"%B %Y"))

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

#merging all dataframes into using datetime indices
total_table = pd.concat([car_size_category,employment_stats,oil_prices,transit_ridership], axis=1)
total_table.to_csv('Merged Total Table.csv', encoding='utf-8', index=False)
