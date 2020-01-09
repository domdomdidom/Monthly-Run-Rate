# Monthly-Run-Rate
Basic monthly run rate that incorporates padding and softening for holidays and data validation.

############################
Author: Dominique Vanden Dries
December 2019
############################

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
+ DIMENSIONS + MEASURES + DATE_COL_NAME + TABLE +
+++++++++++++++++++++++++++++++++++++++++++++++++

HOLIDAY FILE:
