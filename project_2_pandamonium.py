# Import 3rd party libraries
conda install -c anaconda numpy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import local libraries
from threshold_prediction_plot import threshold_prediction_plot, sigmoid

# Configure Notebook
import warnings
warnings.filterwarnings('ignore')


from requests import get
response = get('https://www.goodcarbadcar.net/2019-canada-vehicle-sales-figures-by-model/#monthlysales')

