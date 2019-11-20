import logging
from datetime import date
import pandas as pd
import pathlib
import numpy as np
from run_rate_logic_functions import *
from auto_forecast import *
import sys
pd.options.display.float_format = '{:.5f}'.format

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

def parse_config(config_filepath):
    '''
    use config file to gather and format data
    config file must be either a csv or txt file and include the following columns (not case sensitive)
        - DIMENSIONS (list)
        - MEASURES (list)
        - DATA_LOC (filepath)
        - HOLIDAY_LOC (filepath)
        - DATE_COL_NAME (str)
    ---------------------------------------------------------------------------------------------
    INPUT: filepath to config as sysarg[1]
    OUTPUT: data (dataframe), holidays (dataframe), dimensions (list), measures (list)
    '''

    logger.info("START: readConfig")

    # Check Config Filetype
    if not any(extension in str(config_filepath)[-4:] for extension in ['txt', 'csv']):
        logger.warning("readConfig: config is invalid filetype - .csv or .txt required")
        return
    try:
        config = pd.read_csv(config_filepath, sep = determine_seperator(config_filepath), engine='python')

        if not isinstance(pd.read_csv(config_filepath), pd.core.frame.DataFrame):
            logger.error("readConfig: csv/txt cannot be read as pandas dataframe")
            return
    except:
        logger.error("readConfig: config file not found at " + str(config_filepath))
        return

    config.columns = [x.upper() for x in config.columns] # make everything uppercase
    target_cols = ['DIMENSIONS', 'MEASURES', 'DATA_LOC', 'HOLIDAY_LOC', 'DATE_COL_NAME']

    # Check Validity of Config File
    for col in config.columns:
        if col not in target_cols:
            logger.warning("readConfig: Misspelled or extra column in config file: " + str(col) + str('\n') + str("Valid Column Names (Not Case Sensitive): 'DIMENSIONS', 'MEASURES', 'DATA_LOC', 'HOLIDAY_LOC', 'DATE_COL_NAME'"))
    
    for col in target_cols:
        if col not in config.columns:
            logger.error("readConfig: Missing column from config file: " + str(col) + str('\n') + str("Valid Column Names (Not Case Sensitive): 'DIMENSIONS', 'MEASURES', 'DATA_LOC', 'HOLIDAY_LOC', 'DATE_COL_NAME'"))

    dimensions, measures = [item.strip() for item in config['DIMENSIONS'][0].split(',')], [item.strip() for item in config['MEASURES'][0].split(',')]
    
    logger.info("FINISHED: readConfig")

    # Check and Format Holiday File
    logger.info("START: readHoliday")

    holiday_filepath = pathlib.Path(config['HOLIDAY_LOC'][0])

    if not any(extension in str(holiday_filepath)[-5:] for extension in ['txt', 'csv']):
        logger.warning("readHoliday: 'holiday_calendar' invalid filetype - .csv or .txt required")
        return

    try:
        pd.read_csv(holiday_filepath, sep = determine_seperator(holiday_filepath), engine='python')

    except:
        logger.error("readHoliday: 'holiday_calendar' invalid/mispelled filepath, unable to locate")
        return

    if not isinstance(pd.read_csv(holiday_filepath, sep = determine_seperator(holiday_filepath), engine='python'), pd.core.frame.DataFrame):
        logger.error("readHoliday: File 'holiday_calendar' cannot be read as a dataframe")
        return

    holidays, max_holiday = format_check_holiday_file(pd.read_csv(holiday_filepath, sep = determine_seperator(holiday_filepath), engine='python'))

    if max_holiday <= pd.to_datetime(date.today()):
        logger.warning("readHoliday: Max date approaching. Please update 'holiday_calendar'")

    logger.info("FINISHED: readHoliday")

    # Check data file
    logger.info("START: readData")

    data_filepath = pathlib.Path(config['DATA_LOC'][0])

    if not any(extension in str(data_filepath)[-5:] for extension in ['txt', 'csv']):
        logger.warning("readData: data invalid filetype - .csv or .txt required")
        return

    if not isinstance(pd.read_csv(data_filepath, sep = determine_seperator(data_filepath), engine='python'), pd.core.frame.DataFrame):
        logger.error("readData: data not able to be read as a dataframe")
        return
    
    data = pd.read_csv(data_filepath, sep = determine_seperator(data_filepath), engine='python')

    logger.info("FINISHED: readData")

    # Check all declared dimensions and measures are columns within data
    for dim in dimensions:
        if dim not in data.columns:
            logger.error("readData: DIMENSION present in config file is missing from data file: " + str(dim))
            return

    for meas in measures:
        if meas not in data.columns:
            logger.error("readData: MEASURE present in config file is missing from data file: " + str(meas))
            return
        else:
            if np.issubdtype(data[meas].dtype, np.number) == False:
                logger.warning("readData: MEASURE in data file is non-numeric: " + str(meas))
                return

    # Check datetime
    date_col_name = config['DATE_COL_NAME'][0]

    try:
        data[date_col_name] = pd.to_datetime(data[date_col_name])

    except:
        logger.error("readData: DATE COL in data file invalid date format or does not match DATE_COL_NAME in config: " + str(date_col_name))

    data.rename(columns={date_col_name:'DAY_DATE'}, inplace=True)
    data['DOW'] = pd.to_datetime(data['DAY_DATE']).dt.weekday

    # Returns
    return data, holidays, dimensions, measures

def run_logic(data, holidays, dimensions, measures):

    '''
    use the functions from run_rate_logic_functions to forecast using 7 day merge on 'DOW' and account for holidays
    ---------------------------------------------------------------------------------------------
    INPUT: data dataframe, holiday dataframe, measures, dimensions
    OUTPUT: dataframe with columns = ['DAY_DATE', 'MAPPED_FROM', 'RECORD_TYPE'], MEASURES, DIMENSIONS
    '''

    logger.info("START: Forecast Logic")
    
    logger.info("Fetching date params")
    max_data_day, n_forecast_days = get_time_params(data)

    logger.info("Compiling forecast")
    forecasted = make_forecast(data, max_data_day, n_forecast_days)

    logger.info("Adjusting for holidays")
    holiday_adjusted = holiday_adjust(forecasted, max_data_day, holidays, measures)
    
    logger.info("Formating final output")
    final_out = format_final(holiday_adjusted, max_data_day, dimensions, measures)

    return final_out

def RR_main(argv, check = False):

    data, holidays, dimensions, measures = parse_config(pathlib.Path(argv))
    final_out = run_logic(data, holidays, dimensions, measures)

    if check == True:
        logger.info("START: checkAuto")
        if check_auto('C:\Users\dvdries\python_proj_folder\RR\data\gold_std_compiled_11-19.csv', final_out) == False:
            logger.warning("checkAuto: code got screwed up :(")
        
        else:
            logger.info("checkAuto: passed unit test")

    return final_out

    #### IF WE WANT TO PUSH TO DB IN MAIN PUT CODE HERE

if __name__ == '__main__':

    RR_main(sys.argv[1], check = True)

