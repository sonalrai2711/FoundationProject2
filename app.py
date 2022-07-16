import os
import numpy as np
import pickle
import pandas as pd
from pycaret.regression import *
from datetime import date, datetime, timedelta
import calendar
from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime
from PIL import Image


feature_extractor = load_model(r'consumption-pipeline')

import streamlit as st
import matplotlib.pyplot as plt


def season_calc(month):
    if month in [6, 7, 8, 9, 10]:
        return "Summer"
    else:
        return "Winter"


def get_holidays(dt, holidays):
    if dt.strftime("%Y-%m-%d") in holidays:
        return 1
    else:
        return 0


def getAttributes(sdate, edate):
    # sdate = date(2022, 5, 1)  # start date
    # edate = date(2022, 5, 31)  # end date

    sdge_final_data = pd.read_csv('merged_sdge_data.csv')
    delta = edate - sdate  # as timedelta
    datelist = []
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        datelist.append(day)

    future_df = pd.DataFrame(datelist, columns=['Date'])

    weeknames = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    cal = USFederalHolidayCalendar()

    future_df['Date'] = pd.to_datetime(future_df['Date'])
    future_df['Year'] = future_df.Date.dt.year
    future_df['Month'] = future_df.Date.dt.month
    future_df['Month_Name'] = future_df.Date.dt.month_name()
    future_df['Day'] = future_df.Date.dt.day
    future_df['Week_day'] = future_df.Date.dt.weekday.map(weeknames)
    future_df['Season'] = future_df.Date.dt.month.apply(season_calc)

    holiday_list = []
    holidays = cal.holidays(start=future_df['Date'].min(), end=future_df['Date'].max())

    future_df['Holiday'] = future_df.apply(lambda row: get_holidays(row['Date'], holidays), axis=1)

    # adding in another column which indicates 'non-working' days which include weekends and holidays
    future_df['Non_working'] = future_df.apply(lambda x: 'non-working' if \
        ((x['Holiday'] == 1) or (x['Week_day'] in ['Saturday', 'Sunday']))
    else 'working', axis=1)

    prev_year = future_df['Year'][0] - 1
    DailyCoolingDegreeDays = round(sdge_final_data[(sdge_final_data['Month'] == future_df['Month'][0]) & (
                sdge_final_data['Year'] == prev_year)].loc[:,
                                   'DailyCoolingDegreeDays'].mean(axis=0), 2)
    DailyHeatingDegreeDays = round(sdge_final_data[(sdge_final_data['Month'] == future_df['Month'][0]) & (
                sdge_final_data['Year'] == prev_year)].loc[:,
                                   'DailyHeatingDegreeDays'].mean(axis=0), 2)
    HourlyDryBulbTemperature = round(sdge_final_data[(sdge_final_data['Month'] == future_df['Month'][0]) & (
                sdge_final_data['Year'] == prev_year)].loc[:,
                                     'DailyHeatingDegreeDays'].mean(axis=0), 2)

    future_df['DailyCoolingDegreeDays'] = DailyCoolingDegreeDays
    future_df['DailyHeatingDegreeDays'] = DailyHeatingDegreeDays
    future_df['HourlyDryBulbTemperature'] = HourlyDryBulbTemperature

    prediction = predictor(future_df)
    total_holidays = future_df[(future_df['Holiday'] == 1)].loc[:,'Holiday'].count()
    nonworking_days = future_df[(future_df['Non_working'] == 'non-working')].loc[:,'Non_working'].count()
    htmls = pd.DataFrame({'   Month   ': future_df['Month_Name'][0],'   Season   ': future_df['Season'][0],'   Holidays   ': str(total_holidays), '   Non Working Days   ': str(nonworking_days), '   Prediction   ' : str(prediction)}, index = ['0'])
        
    return htmls


def predictor(data_unseen):
    prediction = predict_model(feature_extractor, data=data_unseen)
    average_comsumption = prediction['Label'].mean()
    return (round(average_comsumption,2))


#Webpage design
#opening the image
image = Image.open('imagefile.jpg')
#displaying the image on streamlit app
st.image(image)

st.title('Electricity Consumption Prediction')


start_date = st.sidebar.date_input('Select the Start Date', datetime.date(2022, 5, 1))


end_date = st.sidebar.date_input('Select the End Date', datetime.date(2022, 5, 31))


# text over upload button "Upload Image"

if st.sidebar.button("Predict"):
    st.write("Energy consumption Prediction between the date range ", start_date, " and ", end_date, " : " )
    #st.write(end_date)
    
    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    if start_date is not None and end_date is not None:
        start_month = datetime.datetime.strptime(str(start_date), "%Y-%m-%d")
        end_month = datetime.datetime.strptime(str(end_date), "%Y-%m-%d")
        html = ''
        html += '<table border="1"><tr><th>Month</th><th>Season</th><th>Holidays</th><th>Non working Days</th>' \
            '<th>Prediction</th></tr>'
        if start_month.month == end_month.month:
            future_df = getAttributes(start_date, end_date)
            st.dataframe(future_df.style.highlight_max(axis=0), width=850)
            #html += future_df
        else:
            target_year = start_month.year
            for m in range(start_month.month, end_month.month+1):
                new_start_date = date(target_year, m, 1)
                new_end_date = date(target_year, m, 30)
                future_df = getAttributes(new_start_date, new_end_date)
                st.dataframe(future_df.style.highlight_max(axis=0),  width=850)

       
