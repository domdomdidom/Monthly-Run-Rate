import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

import sys
import getopt
import pathlib
import cx_Oracle
import pandas
import warnings
import numpy as np
from calendar import *
import datetime
from datetime import timedelta
import pandas as pd
from datetime import date
from pyhive import hive
import ForecastingEnv
pd.options.display.float_format = '{:.5f}'.format

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
        case = 'mid_month'
        
    return max_data_day, case

def validate_forecast_type(df, case, days_so_far, n_days_in_month):
    '''
    determine if sufficent data is available for the type of forecast required (see min_data_dict)
    '''
    min_data_dict = {'first_7_days' : (days_so_far + (7 -  days_so_far)), 'mid_month' : days_so_far, 'EOM' : n_days_in_month}

    if len(set(df['DAY_DATE'])) >= min_data_dict[case]:
        return True
    else:
        return False

def get_time_params(df):
    '''
    establish time parameters based on the case
    ''' 
    max_data_day, case = determine_case(df)

    logger.info("Forecast Logic: case determined as type " + str(case))

    n_days_in_month = monthrange(max_data_day.year, max_data_day.month)[1]
    days_so_far = max_data_day.day
    n_forecast_days = n_days_in_month - days_so_far

    return max_data_day, n_forecast_days, validate_forecast_type(df, case, days_so_far, n_days_in_month)

def make_future_dataframe(max_data_day, n_forecast_days):
    '''
    create a dataframe for the future part of the month using max_data_day and n_forecast_days
    '''
    future_data = pd.DataFrame( [(max_data_day + timedelta(i+1)) for i in range(n_forecast_days)], 
                                 columns = ['CALENDAR_DAY'])

    future_data['DOW'] = pd.to_datetime(future_data['CALENDAR_DAY']).dt.weekday #add in DOW column
    
    return future_data
    
def append_forecast_to_mtd(future_data, last_7, filtered_mtd):
    '''
    merge future_data with last_7 on DOW 
    rename columns to keep track of which are mapped and which are actual
    '''
    future_data = pd.merge(future_data, last_7, on='DOW', how='left').rename(columns = {'DAY_DATE' : 'MAPPED_FROM', 'CALENDAR_DAY' : 'DAY_DATE'})

    forecasted = filtered_mtd.append(future_data, sort=False)
    
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
    '''
    housekeeping - makes dataframe fit the OracleDB schema
    '''
    conditions = [
        df['DAY_DATE'] > pd.to_datetime(max_data_day),
        df['DAY_DATE'] <= pd.to_datetime(max_data_day)]
    choices = ['Forecast', 'Actual']
    
    df['RECORD_TYPE'] = np.select(conditions, choices)

    for col in measures:
        df[col] = df[col].fillna(0)

    final_cols = ['DAY_DATE', 'MAPPED_FROM', 'RECORD_TYPE'] + dimensions + measures

    #more housekeeping
    for date_col in ['DAY_DATE', 'MAPPED_FROM']:
        df[date_col] = pd.to_datetime(df[date_col], errors='ignore').dt.strftime('%Y-%m-%d')

        if date_col == 'MAPPED_FROM':
            df[date_col] = df[date_col].astype(str).replace('NaT', '', regex=True)

    return df[final_cols]
    
def check_auto(gold_std_filepath, df):
    '''
    unit tests function... not working right now
    '''
    gold_std = pd.read_csv(gold_std_filepath, index_col = 'VERTICAL')
    grouped_output = df.groupby('VERTICAL').sum()

    is_equal = gold_std.round(-1).equals(grouped_output.round(-1))
    if is_equal == False:
        print(grouped_output)
        print(gold_std)

    return is_equal

def parse_config(params_dict):
    '''
    use config file to specify measures, dimensions, date column name, Oracle table name
    config file must be either a csv or txt file and include the following columns (not case sensitive)
        - DIMENSIONS (list)
        - MEASURES (list)
        - DATE_COL_NAME (str)
        - TABLE (str)
    ---------------------------------------------------------------------------------------------
    INPUT: dict
    OUTPUT: int, list, list, str, str
    '''
    logger.info("START: readConfig")
    n_errors = 0
    configFile = pathlib.Path(params_dict['configFile'])
    
    try:
        config = pd.read_csv(configFile, sep = determine_seperator(configFile), engine='python')

        if not isinstance(pd.read_csv(configFile), pd.core.frame.DataFrame):
            logger.error("readConfig: csv/txt cannot be read as pandas dataframe")
            n_errors += 1

        config.columns = [x.upper() for x in config.columns] # make everything uppercase
        target_cols = ['DIMENSIONS', 'MEASURES', 'DATE_COL_NAME', 'TABLE']

        # Check Validity of Config File
        for col in config.columns:
            if col not in target_cols:
                logger.warning("readConfig: Misspelled or extra column in config file: " + str(col) + str('\n') + str("Valid Column Names (Not Case Sensitive): 'DIMENSIONS', 'MEASURES', 'DATE_COL_NAME', 'TABLE'"))
        
        for col in target_cols:
            if col not in config.columns:
                logger.error("readConfig: Missing column from config file: " + str(col) + str('\n') + str("Valid Column Names (Not Case Sensitive): 'DIMENSIONS', 'MEASURES', 'DATE_COL_NAME', 'TABLE'"))
                n_errors += 1

        dimensions, measures = [item.strip() for item in config['DIMENSIONS'][0].split(',')], [item.strip() for item in config['MEASURES'][0].split(',')]
        date_col_name = config['DATE_COL_NAME'][0]
        table = config['TABLE'][0]

        if '.' not in table:
            logger.error("readConfig: TABLE needs format 'database.tablename'")

        logger.info("readConfig: DIMENSIONS read from config as " + str(dimensions))
        logger.info("readConfig: MEASURES read from config as " + str(measures))
        logger.info("readConfig: DATE_COL_NAME read from config as " + str(date_col_name))
        logger.info("readConfig: TABLE read from config as " + str(table))
        logger.info("FINISHED: readConfig")

        return n_errors, dimensions, measures, date_col_name, table

    except:
        logger.error("readConfig: config file not found at " + str(configFile))
        n_errors += 1

def parse_holiday(params_dict):
    '''
    read in holidayFile, rename columns, ensure sufficent data, create magnifier columns, return edited dataframe
    ---------------------------------------------------------------------------------------------
    INPUT: dict
    OUTPUT: int, datafame
    '''
    logger.info("START: readHoliday")
    n_errors = 0

    try:
        holidayFile = pathlib.Path(params_dict['holidayFile'])
    
        if not any(extension in str(holidayFile)[-5:] for extension in ['txt', 'csv']):
            logger.error("readHoliday: 'holiday_calendar' invalid filetype - .csv or .txt required")
            n_errors += 1

        pd.read_csv(holidayFile, sep = determine_seperator(holidayFile), engine='python')

        if not isinstance(pd.read_csv(holidayFile, sep = determine_seperator(holidayFile), engine='python'), pd.core.frame.DataFrame):
            logger.error("readHoliday: File 'holiday_calendar' cannot be read as a dataframe")
            n_errors += 1

        holidays, max_holiday = format_check_holiday_file(pd.read_csv(holidayFile, sep = determine_seperator(holidayFile), engine='python'))

        if max_holiday <= pd.to_datetime(date.today()):
            logger.warning("readHoliday: Max date approaching. Please update 'holiday_calendar'")

        logger.info("FINISHED: readHoliday")

        return n_errors, holidays

    except:
        logger.error("readHoliday: 'holiday_calendar' invalid/mispelled filepath, unable to locate")
        n_errors += 1

def parse_data(params_dict, dimensions, measures, date_col_name):
    '''
    execute SQL query on HIVE, validate output against dimensions and measures specified within configFile,
    check for NULLs within query results, format date columns and add DOW col
    ---------------------------------------------------------------------------------------------
    INPUT: dict, list, list, str 
    OUTPUT: int, datafame
    '''
    logger.info("START: readData")
    n_errors = 0

    queryFile = open(str(pathlib.Path(params_dict['queryFile'])), "r")
    query_stmt = queryFile.read()
    queryFile.close()

    hive_conn = hive.connect(host=ForecastingEnv.BIGDATA_SERVER_STRING, 
                              port=ForecastingEnv.BIGDATA_SERVER_PORT,  
                              username=ForecastingEnv.BIGDATA_USER, 
                              auth='LDAP', 
                              password=ForecastingEnv.BIGDATA_PASSWORD,
                              database='default')
    hive_curs = hive_conn.cursor()

    try: 
        data = pd.read_sql(query_stmt, hive_conn)
        data.columns = map(str.upper, data.columns)
        logger.info("readData: query executed successfully. Returned dataframe of row length = " + str(len(data)))
        hive_curs.close()
        hive_conn.close()
        logger.info("readData: " + str(data.info()))

        if len(data) <= 1:
            logger.error("readData: empty Dataframe")
            n_errors += 1

    except:
        logger.error("readData: query failure")
        n_errors += 1
        hive_curs.close()
        hive_conn.close()
    
    # Validate data using dimensions, measures, date_col_name
    logger.info("readData: Starting data validation against config file")
    for dim in dimensions:
        if dim not in data.columns:
            logger.error("readData: DIMENSION present in config file is missing from data file (check case): " + str(dim))
            n_errors += 1

        else:
            if data[dim].isnull().values.any():
                logger.warning("readData: DIMENSION contains NULL values. Consider reviewing this col: " + str(dim))

    for meas in measures:
        if meas not in data.columns:
            logger.error("readData: MEASURE present in config file is missing from data file: (check case)" + str(meas))
            n_errors += 1

        else:
            if np.issubdtype(data[meas].dtype, np.number) == False:
                logger.warning("readData: MEASURE in data file is non-numeric: " + str(meas))
    try:
        data[date_col_name] = pd.to_datetime(data[date_col_name])

    except:
        logger.error("readData: DATE COL in data file invalid date format or does not match DATE_COL_NAME in config: " + str(date_col_name))
        n_errors += 1

    data.rename(columns={date_col_name:'DAY_DATE'}, inplace=True)
    data['DOW'] = pd.to_datetime(data['DAY_DATE']).dt.weekday

    logger.info("FINISHED: readData")

    return n_errors, data

def check_complete(holidays, data, measures):
    '''
    normalize data for holidays (if there were holidays prior, artificially increase the data as if there were none)
    consider all data where DOW is the same as max_data_day. calculate deltas for all measures between max_data_day and prior data (i.e. all available Tuesdays)
    let x = smallest NEGATIVE delta (which suggests a lack of data completeness for max_data_day, we don't care if there is a positive delta)
    warn if x exceeds a pct_dif of 40%, fail if x exceeds pct_dif of 60%
    ---------------------------------------------------------------------------------------------
    INPUT: dataframe, dataframe, list
    OUTPUT: int
    '''

    logger.info("START: checkComplete")
    n_errors = 0

    max_data_day = data['DAY_DATE'].max()

    max_data = data[(data['DAY_DATE'] == max_data_day)]
    comp_data = data[(data['DAY_DATE'] != max_data_day) & (data['DOW'] == max_data['DOW'].values[0])]

    max_hol = holidays.loc[holidays['DAY_DATE'] == max_data_day]
    comp_hol = holidays.loc[holidays['DAY_DATE'].isin(set(comp_data['DAY_DATE']) & set(holidays['DAY_DATE']))]

    max_mrg = pd.merge(max_data, max_hol, how='left').fillna({'magnifier':1})
    comp_mrg = pd.merge(comp_data, comp_hol, how='left').fillna({'magnifier':1})

    for df in [max_mrg, comp_mrg]:
        for meas in measures:
            df[meas] = df[meas] * df['magnifier']
        
    max_df = max_mrg.groupby('DAY_DATE').sum()
    comp_df = comp_mrg.groupby('DAY_DATE').sum()

    delta_df = pd.DataFrame()
    for i, row in comp_df.iterrows():
        delta_df = delta_df.append(row-max_df)

    dif_dict = delta_df[delta_df[measures] > 0].min().to_dict()

    for meas in measures:
    
        max_val = max_df[meas][0]
        comp_val = max_val + dif_dict[meas]
        
        dif = np.abs((max_val-comp_val)/comp_val)
        
        if dif >= .40:
            logger.warning("checkComplete: Possible missing data. There is a delta of %" 
                            + str(dif) + " in measure: " + str(meas))
            if dif >= .60:
                n_errors += 1
                logger.error("checkComplete: FAILURE due to possible missing data. There is a delta of %" 
                            + str(dif) + " in measure: " + str(meas))

    logger.info("FINISHED: checkComplete")
    return n_errors

def basic_monthlyForecast(data, holidays, dimensions, measures):
    '''
    use above functions to forecast
    ---------------------------------------------------------------------------------------------
    INPUT: dataframe, dataframe, list, list
    OUTPUT: int, dataframe
    '''
    logger.info("START: Forecast Logic")
    n_errors = 0
    
    logger.info("Forecast Logic: Fetching date params")
    max_data_day, n_forecast_days, valid_data_bool = get_time_params(data)

    if valid_data_bool == False:
        logger.error("Forecast Logic: insufficent data submitted for forecast type")
        n_errors += 1

    logger.info("Forecast Logic: Compiling forecast")
    forecasted = make_forecast(data, max_data_day, n_forecast_days)

    logger.info("Forecast Logic: Adjusting for holidays")
    holiday_adjusted = holiday_adjust(forecasted, max_data_day, holidays, measures)
    
    logger.info("Forecast Logic: Formating final output")
    final_out = format_final(holiday_adjusted, max_data_day, dimensions, measures)
    
    return n_errors, final_out

#dataframe: A pandas dataframe containing the rows to be used in the sql_query
#columns: A list with the column names in order to be used by the sql_query statement
#initial_statement: A statement to be executed prior to the insert (useful to clear old data)
#insert_statement: A string containing the insert_statement to execute. Positional parameters in the form :1, :2, etc should be used.
#                  At runtime those positional parameters will be populated with the contents of the columns (in the order specified by columns)
# updateBatchSize: By default 500, it specifies how many updates to pool before sending them to the DB. A value of 0 means all at once.

def pushDataToDB(dataframe, columns, initial_statement_DWHS, insert_statement_DWHS, updateBatchSize = 500):
    logger.info("START: pushDataToDB")

    try:
        dwhs_conn = cx_Oracle.Connection(ForecastingEnv.QMPX_CONNECT_STRING)
        dwhs_curs = dwhs_conn.cursor()

        logger.info("Prepare data for DWHS update")
        updatesList = dataframe.loc[:,columns].to_records(index=False).tolist()
        logger.info(str(len(updatesList)))

        logger.info("Executing initial statement")
        dwhs_curs.execute(initial_statement_DWHS)

        elementCount = len(updatesList)
        logger.info("Will insert %d rows", elementCount)

        #If updateBatchSize == 0 then insert all records in one block
        if updateBatchSize==0:
            updateBatchSize=elementCount

        startIndex = 0
        endIndex = min(startIndex + updateBatchSize,elementCount)
        while(startIndex < elementCount):
            rows = updatesList[startIndex:endIndex]
            dwhs_curs.executemany(insert_statement_DWHS,rows)
            #logger.info("Execute statement for insert_statement finished, rows[%d,%d]", startIndex,endIndex)
            startIndex = endIndex #CHANGING THIS USED TO BE ENDINDEX + 1
            endIndex = min(startIndex + updateBatchSize,elementCount)

        logger.info("Finished updating DB")

        dwhs_conn.commit()

        logger.info("Closing DB connection")
        dwhs_curs.close()
        dwhs_conn.close()

    except cx_Oracle.DatabaseError as error:
        logger.error("Oracle-Error: %s", error)
        raise
    
def print_help(params):
    logger.info("params=%s",params)

def main(argv):    

    params_dict={"pushToDB":"", "configFile":"", "queryFile":"", "holidayFile":""}

    try:
            params_read, args = getopt.getopt(argv,"h",[param+"=" for param in params_dict])
    except getopt.GetoptError as e:
            logger.info("getopt.GetoptError=%s",e)
            print_help(params_dict)
            sys.exit(2)
    for param, value in params_read:
            if param == '-h':
                print_help()
                sys.exit()
            elif param[2:] in params_dict:
                params_dict[param[2:]] = value

    for param, value in params_dict.items():
        if (value==""):
            logger.info("Missing parameter '%s'",param)
            print_help(params_dict)
            sys.exit(2)

    # Dom Code
    n_errors, dimensions, measures, date_col_name, table = parse_config(params_dict)
    if n_errors > 0:
        logger.error("Breaking on error in parse_config") 
        return
    n_errors, holidays = parse_holiday(params_dict)
    if n_errors > 0:
        logger.error("Breaking on error in parse_holiday") 
        return
    n_errors, data = parse_data(params_dict, dimensions, measures, date_col_name)
    if n_errors > 0:
        logger.error("Breaking on error in parse_data") 
        return
    n_errors = check_complete(holidays, data, measures)
    if n_errors > 0:
        logger.error("Breaking on error checkComplete") 
        return
    n_errors, dataframe = basic_monthlyForecast(data, holidays, dimensions, measures)
    if n_errors > 0:
        logger.error("Breaking within forecast. The data provided may be insufficent for the type of forecast requested, or there is a data delay.") 
        return

    #Global settings
    pandas.set_option('expand_frame_repr', False)
    pandas.options.display.float_format = '{:.5f}'.format
    logger.info("Params %s", params_dict)

    #Sample call
    columns = ['DAY_DATE', 'MAPPED_FROM', 'RECORD_TYPE']
    columns.extend(dimensions)
    columns.extend(measures)

    start_val, end_val = int((len(columns)) - (len(dimensions) + len(measures))), int(len(columns) + 1)
    binds = [str(':') + str(num) for num in range(start_val, end_val)]

    initial_statement_DWHS = """DELETE FROM """ + str(table) + """ WHERE FORECAST_UPLOAD_TRUNC_DATE = TRUNC(SYSDATE)"""
    insert_statement_DWHS = """INSERT INTO """ + str(table) + """ (FORECAST_UPLOAD_TRUNC_DATE, \n UPDATED_DATETIME, \n """ + str(', \n '.join(columns)) + """) \n """ + """ VALUES (TRUNC(SYSDATE), SYSDATE, TO_DATE(:1,'YYYY-MM-DD'), TO_DATE(:2,'YYYY-MM-DD'), """ + str(', '.join(binds)) + """)"""

    pushDataToDB(dataframe, columns, initial_statement_DWHS, insert_statement_DWHS)
    

if __name__ == "__main__":
    main(sys.argv[1:])
