# Monthly-Run-Rate
Basic monthly run rate that incorporates temporal patterns for holidays based on previous data. Data is pulled off HIVE server. Script includes a data validation function to prevent it from running if data from yesterday isn't fully loaded yet. 


Author: Dominique Vanden Dries

December 2019


PYTHON FILES:

1. auto_forecast
	logical RR functions, parse flags and config functions, query Hive functions,  pushtoDB functions
2. unit_tests
	validate code if changed
3. ForecastingEnv
	DWHS write authentication credentials, Hive credentials

SQL FILES:
- SQL query to be executed by HIVE. Can be a .txt, .sql, etc file. uses with open("r")

CONFIG FILE:

+++++++++++++++++++++++++++++++++++++++++++++++++
DIMENSIONS + MEASURES + DATE_COL_NAME + TABLE
+++++++++++++++++++++++++++++++++++++++++++++++++


HOLIDAY FILE:

+++++++++++++++++++++++++++++++++++++++++++++++++
DAY_DATE + DAY_COMMENTS + HOLIDAY_PCT_MODIFIER
+++++++++++++++++++++++++++++++++++++++++++++++++

SAMPLE CMD LINE:


python auto_forecast.py 
--pushToDB=True 
--configFile=/opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Forecasting/src/config_auto_forecast_temp.cfg 
--holidayFile=/opt/home/dvdries/dvdries_abacus/perforce/dev/QMPX/models/TemporalPatterns/holiday_calendar.txt 
--queryFile=/opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Forecasting/src/TEMP_RR_query.sql

