import warnings
import numpy as np
from calendar import *
import datetime
from datetime import timedelta
import pandas as pd

def determine_seperator(filepath):
    if 'csv' in str(filepath)[-5]:
        seperator = ','
    elif 'txt' in str(filepath)[-5]:
        seperator = '\t'
    else:
        return
    return seperator

def format_check_holiday_file(dataframe):
    dataframe['DAY_DATE'] = pd.to_datetime(dataframe['DAY_DATE']).dt.strftime('%m/%d/%Y')
    dataframe.rename(columns={'HOLIDAY_PCT_MODIFIER':'softener'}, inplace=True)
    dataframe['magnifier'] = 1./dataframe['softener']
    return dataframe, pd.to_datetime(dataframe['DAY_DATE'].values[-8])

def determine_case(df):
    '''
    use the max_data_day to determine which timeframe we are examining (EOM, normal, first_7_days)
    '''
    
    max_data_day = df['DAY_DATE'].max()
    
    if (max_data_day + timedelta(1)).day == 1:
        case = 'EOM'
    elif 1 < max_data_day.day < 8:
        case = 'first_7_days'
    else:
        case = 'normal'
        
    print(str('case: ' + str(case)))
    return max_data_day, case

def get_time_params(df):
    '''
    establish time parameters based on the case
    ''' 
    max_data_day, case = determine_case(df)

    n_days_in_month = monthrange(max_data_day.year, max_data_day.month)[1]
    days_so_far = max_data_day.day
    n_forecast_days = n_days_in_month - days_so_far
    
    return max_data_day, n_forecast_days #, days_so_far, n_days_in_month

def make_future_dataframe(max_data_day, n_forecast_days):
    '''
    create a dataframe for the future part of the month using max_data_day and n_forecast_days
    '''
    
    future_data = pd.DataFrame( [(max_data_day + timedelta(i+1)) for i in range(n_forecast_days)], 
                                 columns = ['CALENDAR_DAY'])

    future_data['DOW'] = future_data['CALENDAR_DAY'].dt.weekday #add in DOW column
    
    return future_data
    
def append_forecast_to_mtd(future_data, last_7, filtered_mtd):
    '''
    merge future_data with last_7 on DOW 
    rename columns to keep track of which are mapped and which are actual
    '''
    
    future_data = pd.merge(future_data, last_7, on='DOW', how='left').rename(columns = {'DAY_DATE' : 'MAPPED_FROM', 'CALENDAR_DAY' : 'DAY_DATE'})

    forecasted = filtered_mtd.append(future_data)
    
    return forecasted

def make_forecast(df, max_data_day, n_forecast_days):
    '''
    this function lives under a nested for-loop to iterate through all combinations of vertical/channel
    '''

    df['DOW'] = pd.to_datetime(df['DAY_DATE']).dt.weekday

    # filtered_mtd will *only* show the RR month's data, no data from previous month. We use this as base to build the forecast 
    filtered_mtd = df[df['DAY_DATE'].apply(lambda x: x.month) == max_data_day.month]
    
    last_7 = df[(df['DAY_DATE'] >= (max_data_day - timedelta(6)))] # last 7 days of data, regardless of month

    future_data = make_future_dataframe(max_data_day, n_forecast_days)

    forecasted = append_forecast_to_mtd(future_data, last_7, filtered_mtd) # forecasted for entire month
    
    return forecasted

def holiday_adjust(df, max_data_day, holidays, measures):
    '''
    use the holidays csv to apply softening and magnifying 
    '''

    # make dt format match that of the holidays csv.... a little ugly
    change_times = ['DAY_DATE', 'MAPPED_FROM']

    for col in change_times:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%m/%d/%Y')

    # create/rename softener and magnifier columns (mag = 1/soft)
    soft = holidays[['softener', 'DAY_DATE']]
    mag = holidays[['magnifier', 'DAY_DATE']].rename(columns={'DAY_DATE':'MAPPED_FROM'})

    # merge softening with DAY_DATE and magnifying with MAPPED_FROM
    merged1 = pd.merge(df, soft, on = 'DAY_DATE', how='left')
    merged2 = pd.merge(merged1, mag, on = 'MAPPED_FROM', how='left')

    # fill the nulls from the merge with 1
    merged2[['softener', 'magnifier']] = merged2[['softener', 'magnifier']].fillna(value = 1)

    # create a binary col for forecast & multiply soft/mag/bool
    merged2['DAY_DATE'] = pd.to_datetime(merged2['DAY_DATE'])
    merged2['forecast_binary'] = merged2['DAY_DATE'] >= max_data_day
    merged2['multiplier_col_vector'] = (merged2['softener'] * merged2['magnifier'] * merged2['forecast_binary']).replace({0:1})

    # multiply accross and overwrite column
    for col in measures:
        merged2[col] = merged2[col] * merged2['multiplier_col_vector'] 
        
    return merged2

def format_final(df, max_data_day, dimensions, measures):

    conditions = [
        df['DAY_DATE'] > pd.to_datetime(max_data_day),
        df['DAY_DATE'] <= pd.to_datetime(max_data_day)]
    choices = ['Forecast', 'Actual']
    
    df['RECORD_TYPE'] = np.select(conditions, choices)

    for col in measures:
        df[col] = df[col].fillna(0)

    final_cols = ['DAY_DATE', 'MAPPED_FROM', 'RECORD_TYPE'] + dimensions + measures
    return df[final_cols]

def check_auto(gold_std_filepath, df):

    gold_std = pd.read_csv(gold_std_filepath, index_col = 'VERTICAL')
    grouped_output = df.groupby('VERTICAL').sum()

    rez = gold_std.round(-1).equals(grouped_output.round(-1))
    if rez == False:
        print(grouped_output)
        print(gold_std)

    return rez

