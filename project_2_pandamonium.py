# Import 3rd party libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# Import local libraries
#from threshold_prediction_plot import threshold_prediction_plot, sigmoid

# Configure Notebook
import warnings
warnings.filterwarnings('ignore')


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
data_2019 = pd.DataFrame({'model_name': model_names, 'January': january,
                          'February': february, 'March': march,'April': april,
                          'May':may, 'June':june, 'July':july,'August':august,
                          'September': september, 'October': october,
                          'November': november,'December': december})

monthly_sums_2019 = list(data_2019.sum(axis = 0)[1:])

print(data_2019,monthly_sums_2019)

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
data_2020 = pd.DataFrame({'model_name': model_names, 'January': january,
                          'February': february, 'March': march,'April': april,
                          'May':may, 'June':june, 'July':july,'August':august,
                          'September': september, 'October': october,
                          'November': november,'December': december})

monthly_sums_2020 = list(data_2020.sum(axis = 0)[1:])

print(data_2020,monthly_sums_2020)

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
data_2021 = pd.DataFrame({'model_name': model_names, 'January': january,
                          'February': february, 'March': march,'April': april,
                          'May':may, 'June':june, 'July':july,'August':august,
                          'September': september, 'October': october,
                          'November': november,'December': december})

monthly_sums_2021 = list(data_2021.sum(axis = 0)[1:])

print(data_2021,monthly_sums_2021)