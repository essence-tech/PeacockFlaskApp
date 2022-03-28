
##---------------------------------------------------------------------------------------------------------------------

from crypt import methods
from flask import Flask, render_template, url_for, request
app = Flask(__name__)

##---------------------------------------------------------------------------------------------------------------------
# Model
##---------------------------------------------------------------------------------------------------------------------

import base64
from io import BytesIO

import pandas as pd
import numpy as np
import random
from matplotlib.figure import Figure
from datetime import date, datetime

import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import *  # train_test_split, GridSearchCV
from sklearn.inspection import *  # permutation importance
from sklearn.metrics import * # mean_squared_error

import lightgbm
from lightgbm import LGBMRegressor

import pickle

import optuna

##############################################
# User Defined Functions

# Define the function to return the SMAPE value
def symmetric_mean_absolute_percentage_error(actual, predicted) -> float:
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)

    return np.round(
        np.mean(
            np.abs(predicted - actual) /
            ((np.abs(predicted) + np.abs(actual)) / 2)
        ) * 100, 2
    )

def unique_list(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

##############################################
# Load Data

DATA_URL1 = ('20220124_accountFlowNetAddCheck.csv')
DATA_URL2 = ('20220310_ResponseVariableFormated.csv')
DATA_URL3 = ('20220325_WPFormated.csv')

def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data

#####################################
# Load the data into the dataframe
acctDF = load_data(DATA_URL1)
datDF = load_data(DATA_URL2)
wpDFMG = load_data(DATA_URL3)

#####################################
# Pre-process AccountFlow data
acctDF = acctDF[(acctDF['Date']>='2021-01-01') & (acctDF['Date']<='2021-12-31')].reset_index(drop=True)
acctDF['Date'] = pd.to_datetime(acctDF['Date'])
acctDF = acctDF.sort_values(by=['Date'])
acctDF.index = acctDF.pop('Date')

#####################################
# Create dataframe for each account entitlement
freeDF = acctDF[(acctDF['Account_Entitlement']=='Free') & (acctDF['paying_account_flag']=="NonPaying")]
freeDF = freeDF.resample("D").sum()

premiumPayDF = acctDF[(acctDF['Account_Entitlement']=='Premium') & (acctDF['paying_account_flag']=="Paying")]
premiumPayDF = premiumPayDF.resample("D").sum()

premiumNonPayDF = acctDF[(acctDF['Account_Entitlement']=='Premium') & (acctDF['paying_account_flag']=="NonPaying")]
premiumNonPayDF = premiumNonPayDF.resample("D").sum()

premiumPlusPayDF = acctDF[(acctDF['Account_Entitlement']=='Premium+') & (acctDF['paying_account_flag']=="Paying")]
premiumPlusPayDF = premiumPlusPayDF.resample("D").sum()

#####################################
# Pre-process response variable data
datDF = datDF[(datDF['Date']>='2021-01-01') & (datDF['Date']<='2021-12-31')].reset_index(drop=True)
datDF['Date'] = pd.to_datetime(datDF['Date'])
datDF = datDF.sort_values(by=['Date'])
datDF.index = datDF.pop('Date')

#####################################
# Pre-process WP data
wpDFMG = wpDFMG[(wpDFMG['Date']>='2021-01-01') & (wpDFMG['Date']<='2021-12-31')].reset_index(drop=True)
wpDFMG['Date'] = pd.to_datetime(wpDFMG['Date'])
wpDFMG = wpDFMG.sort_values(by=['Date'])
wpDFMG.index = wpDFMG.pop('Date')

#####################################
# Create dataset and add WP columns to dataframe

datasetDF = datDF.copy()

for col in wpDFMG.columns:
  datasetDF[col] = wpDFMG[col]

option1 = 'Free Non-Pay'

if option1 == 'Free Non-Pay':
    datasetDF['Inflow'] = freeDF['Inflows']
    datasetDF['Outflow'] = freeDF['Outflows']*-1

elif option1 == 'Premium Pay':
    datasetDF['Inflow'] = premiumPayDF['Inflows']
    datasetDF['Outflow'] = premiumPayDF['Outflows']*-1

elif option1 == 'Premium Non-Pay':
    datasetDF['Inflow'] = premiumNonPayDF['Inflows']
    datasetDF['Outflow'] = premiumNonPayDF['Outflows']*-1
elif option1 == 'Premium Plus Pay':
    datasetDF['Inflow'] = premiumPlusPayDF['Inflows']
    datasetDF['Outflow'] = premiumPlusPayDF['Outflows']*-1
else:
    pass

datasetCOLS = list(datasetDF.columns)

# Remove Impression columns
spendCOLS = []
for i in range(len(datasetCOLS)):
    if datasetCOLS[i].split("_")[0] == 'Impressions':
        pass
    else:
        spendCOLS.append(datasetCOLS[i])

# Filter dataframe and subset based on time interval
dataset = datasetDF[spendCOLS].copy()
dataset = dataset.reset_index(drop=False)
dataset = dataset[(dataset['Date']>='2021-01-01') & (dataset['Date']<='2021-09-30')]

# Add dummy variables to dataframe
dataset['Month'] = [dataset['Date'][i].month for i in range(len(dataset))]
dataset['DOW'] = [dataset['Date'][i].dayofweek for i in range(len(dataset))]
dataset['DOY'] = [dataset['Date'][i].dayofyear for i in range(len(dataset))]
dataset['fracDOY'] = [dataset['Date'][i].dayofyear/365 for i in range(len(dataset))]

# Remove date column
date_time = dataset.pop('Date')

responseVarLST = spendCOLS
responseVarLST.append('Date')

plot_x_selections = {"x-value": responseVarLST}

predictorVarLST = spendCOLS

plot_y_selections = {"y-value": predictorVarLST}

##---------------------------------------------------------------------------------------------------------------------
# Route calls
##---------------------------------------------------------------------------------------------------------------------

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/visMode", methods=["POST", "GET"])
def visMode():
    if request.method == "POST":
        tierValue = request.form["tierValue"]
        var0 = request.form["xValue"]
        var1 = request.form["yValue"]

        plotDF = datDF.copy()

        for col in wpDFMG.columns:
            plotDF[col] = wpDFMG[col]

        if tierValue == 'Free Non-Pay':
            plotDF['Inflow'] = freeDF['Inflows']
            plotDF['Outflow'] = freeDF['Outflows']*-1

        elif tierValue == 'Premium Pay':
            plotDF['Inflow'] = premiumPayDF['Inflows']
            plotDF['Outflow'] = premiumPayDF['Outflows']*-1

        elif tierValue == 'Premium Non-Pay':
            plotDF['Inflow'] = premiumNonPayDF['Inflows']
            plotDF['Outflow'] = premiumNonPayDF['Outflows']*-1

        elif tierValue == 'Premium Plus Pay':
            plotDF['Inflow'] = premiumPlusPayDF['Inflows']
            plotDF['Outflow'] = premiumPlusPayDF['Outflows']*-1
        else:
            pass

        datasetCOLS = list(plotDF.columns)

        # Remove Impression columns
        spendCOLS = []
        for i in range(len(datasetCOLS)):
            if datasetCOLS[i].split("_")[0] == 'Impressions':
                pass
            else:
                spendCOLS.append(datasetCOLS[i])

        # Filter dataframe and subset based on time interval
        datasetPlot = plotDF[spendCOLS].copy()
        datasetPlot = datasetPlot.reset_index(drop=False)
        datasetPlot = datasetPlot[(datasetPlot['Date']>='2021-01-01') & (datasetPlot['Date']<='2021-09-30')]

        # Remove date column
        date_time = datasetPlot.pop('Date')

        dateFlag = 0
        if var0 == 'Date':
            var0 = date_time
            dateFlag = 1
        
        fig = Figure()
        ax = fig.subplots()
        if dateFlag == 1:
            ax.plot(var0, datasetPlot[var1], 'o')
            ax.set_xlabel('Date', fontsize=16)
            ax.set_ylabel(var1, fontsize=16)
        else:
            ax.plot(datasetPlot[var0], datasetPlot[var1], 'o-')
            ax.set_xlabel(var0, fontsize=16)
            ax.set_ylabel(var1, fontsize=16)
        ax.grid(True)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return render_template('visMode.html', 
                                plot_x_selections=plot_x_selections, 
                                plot_y_selections=plot_y_selections, 
                                data=data)
    else:
        return render_template('visMode.html', plot_x_selections=plot_x_selections, plot_y_selections=plot_y_selections)

@app.route("/feedbackMode")
def feedbackMode():
    return render_template('feedbackMode.html')


if __name__ == '__main__':
    app.run(debug=True)