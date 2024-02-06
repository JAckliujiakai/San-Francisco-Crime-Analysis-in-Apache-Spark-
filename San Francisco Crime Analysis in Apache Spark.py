# Databricks notebook source
# DBTITLE 1,Import package 

from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"
#####################


# COMMAND ----------

# 从SF gov 官网读取下载数据
#import urllib.request
#urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
#dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))
## 或者自己下载
# https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD


# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"
# use this file name later

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql

#initiate sparksession

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

#build a spark frame

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
# df_opt1.show()
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

type(df_opt1)

# COMMAND ----------

# DBTITLE 1,Q1 counts the number of crimes for different category


# COMMAND ----------

# #counts the number of crimes for different category.
# spark.sql(string of SQL)
spark_sql_q1 = spark.sql("SELECT category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
# spark_sql_q1 is spark Dataframe
display(spark_sql_q1)
# job -> stage -> task  

# COMMAND ----------

# DBTITLE 1,data frame based solution
#Spark SQL based
# df way
spark_df_q1 = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(spark_df_q1)

# df_opt1 spark Dataframe groupby()-> GroupedData
# GroupedData count()-> sparkDataframe
# sparkDataframe orderBy()-> sparkDataframe
# spark_df_q1 is sparkDataframe


# COMMAND ----------

# DBTITLE 1,Visualize results
display(spark_df_q1)

# COMMAND ----------

# sql way
spark_sql_q1 = spark.sql("SELECT category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(spark_sql_q1)

# COMMAND ----------

#pandas visulization-pandas creation
# spark_sql_q1 is sparkDataframe 
# toPandas() -> pandas dataframe
crimes_pd_df = spark_sql_q1.toPandas()
display(crimes_pd_df)
print (type(crimes_pd_df))

# COMMAND ----------

##1 Pandas visulization-Plot
plt.figure()
#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
ax = crimes_pd_df.plot(kind = 'bar',x = 'category',y = 'Count',logy= True,color = 'pink',legend = False, align = 'center')
ax.set_ylabel('count',fontsize = 12)
ax.set_xlabel('category',fontsize = 12)
plt.xticks(fontsize=5, rotation=90)
plt.title('#1 Number of crimes for different categories')
display()

# COMMAND ----------

# df way
spark_df_q1 = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(spark_df_q1)

# COMMAND ----------

# MAGIC %md
# MAGIC Q2: Counts the number of crimes for different district, and visualization
# MAGIC

# COMMAND ----------

# sql way
spark_sql_q2 = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY 1 ORDER BY 2 DESC")
display(spark_sql_q2)

# COMMAND ----------

# df way
spark_df_q2 = df_opt1.groupBy('PdDistrict').count().orderBy('Count', ascending=False)
display(spark_df_q2)

# COMMAND ----------

# Data Visulization 
crimes_dis_pd_df = spark_sql_q2.toPandas()
plt.figure()

ax = crimes_dis_pd_df.plot(kind = 'bar',x='PdDistrict',y = 'Count',logy= True,color = 'pink',legend = False, align = 'center')
ax.set_ylabel('count',fontsize = 12)
ax.set_xlabel('district',fontsize = 12)
plt.xticks(fontsize=8, rotation=30)
plt.title('#2 Number of crimes for different districts')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Q3: Count the number of crimes each "Sunday" at "SF downtown".   
# MAGIC assume SF downtown spacial range: X (-122.4213,-122.4313), Y(37.7540,37.7740)
# MAGIC  

# COMMAND ----------

# MAGIC %md method1: spark sql

# COMMAND ----------

q3_result = spark.sql("""
                      with Sunday_dt_crime as(
                      select substring(Date,1,5) as Date,
                             substring(Date,7) as Year
                      from sf_crime
                      where (DayOfWeek = 'Sunday'
                             and -122.423671 < X
                             and X < 122.412497
                             and 37.773510 < Y
                             and Y < 37.782137)
                             )
                             
                      select Year, Date, COUNT(*) as Count
                      from Sunday_dt_crime
                      group by Year, Date
                      order by Year, Date
                      """)
display(q3_result)

# COMMAND ----------

# MAGIC %md method2: dataframe

# COMMAND ----------

df_opt2 = df_opt1[['IncidntNum', 'Category', 'Descript', 'DayOfWeek', 'Date', 'Time', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'Location']]
display(df_opt2)
df_opt2.createOrReplaceTempView("sf_crime")

# COMMAND ----------

from pyspark.sql.functions import hour, date_format, to_date, month, year
# add new columns to convert Date to date format
df_new = df_opt2.withColumn("IncidentDate",to_date(df_opt2.Date, "MM/dd/yyyy")) 
# extract month and year from incident date
df_new = df_new.withColumn('Month',month(df_new['IncidentDate']))
df_new = df_new.withColumn('Year', year(df_new['IncidentDate']))
display(df_new.take(5))

# COMMAND ----------

# df way
sf_downtown = (df_new.X > -122.4313) & (df_new.X < -122.4213) & (df_new.Y < 37.7740) & (df_new.Y > 37.7540 )
spark_df_q3 = df_new.filter((df_new.DayOfWeek == "Sunday") & (sf_downtown)).groupby('IncidentDate','DayOfWeek').count().orderBy('IncidentDate')

# COMMAND ----------

display(spark_df_q3)

# COMMAND ----------

# MAGIC %md
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

# MAGIC %md ###### Q4_Spark Dataframe Way

# COMMAND ----------

df_opt2.createOrReplaceTempView('sf_crime')


# COMMAND ----------

# substring(string, start, length)
# difference between where and having
# where on: row's data (before groupby)
# having on: on aggregated data (after groupby)
spark_sql_q4 = spark.sql("""
                       SELECT SUBSTRING(Date,1,2) AS Month, SUBSTRING(Date,7,4) AS Year, COUNT(*) AS Count
                       FROM sf_crime
                       GROUP BY Year, Month
                       HAVING Year in (2015, 2016, 2017, 2018) 
                       ORDER BY Year, Month
                       """)
display(spark_sql_q4)


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT SUBSTRING(Date,1,2) AS Month, SUBSTRING(Date,7,4) AS Year,COUNT(*) AS Count
# MAGIC FROM sf_crime
# MAGIC GROUP BY Year, Month
# MAGIC HAVING Year in (2015, 2016, 2017, 2018) 
# MAGIC ORDER BY Year, Month
# MAGIC
# MAGIC --可由spark sql格式替换为%sql格式
# MAGIC

# COMMAND ----------


# Show the number of larceny/theft in each month in 2015-2018
q4_result = spark.sql("""
                      with Monthly_crime as(
                      select Date,
                             substring(Date,7) as Year,
                             substring(Date,1,2) as Month
                      from sf_crime
                      where Category = 'LARCENY/THEFT'
                      )
                      
                      select Year, Month, COUNT(*) as Count
                      from Monthly_crime
                      where Year in ('2015', '2016', '2017', '2018')
                      group by Year, Month
                      order by Year, Month
                      """)
display(q4_result)

# COMMAND ----------

years = [2015, 2016, 2017, 2018]
df_years = df_new[df_new.Year.isin(years)]
display(df_years.take(5))

# COMMAND ----------

spark_df_q4 = df_years.groupby(['Year', 'Month']).count().orderBy('Year','Month')
display(spark_df_q4)

# COMMAND ----------

##4 Define udf
def date_to_month(x):
  m_y = [x.split('/')[0],'/',x.split('/')[2]]
  return ''.join(m_y)

def date_to_year(x):
  y = [x.split('/')[2]] 
  return int(''.join(y))

dtm_udf = udf(lambda x: date_to_month(x))
dty_udf = udf(lambda x: date_to_year(x))

# def is_sf_downdown():
# register udf -> apply to spark dataframe

# python function -> def date_to_month -> '09/28/2003' -> '09/2003' 
# lamda fuction -> lambda x: date_to_month
# udf() -> lamda function used as input argument for udf call
# udf -> dtm_udf
# apply udf -> df_opt1.withColumn('Month', dtm_udf('Date'))

# COMMAND ----------

##4 groupby month after 2014
# df_month = df_opt1.select('Date',dtm_udf('Date').alias('Month'))
# similar to the usage of to_date, month, year functions
df_opt2 = df_opt1.withColumn('Month',dtm_udf('Date')) # add a new column with date turned into month
df_opt3 = df_opt2.withColumn('Year',dty_udf('Date')) # add a new column with date turned into month
df_month_res = df_opt3.filter(df_opt3.Year>2014).groupBy('Month').count().orderBy('Count')
display(df_month_res)

# COMMAND ----------

##4 Visulize the result
df_month_res_pd_df = df_month_res.toPandas()
plt.figure()
ax = df_month_res_pd_df.plot(kind ='bar',x='Month',y='count',color = 'pink',legend = False)
ax.set_ylabel('count',fontsize = 12)
ax.set_xlabel('month',fontsize = 12)
plt.xticks(fontsize=6, rotation=70)
plt.title('#4 Total crimes in different months from 2015-2018')
display()

# COMMAND ----------

# MAGIC %md Q4_Business Impact:
# MAGIC It is very obvious from the above figure that the crime rate from 2015 to 2017 is very high, especially the theft crime, and there has been a downward trend in 2018, especially in May.
# MAGIC The crime rate has been so high since 2015, it may be because of the 47th Act signed by the governor in the California referendum in 2014, which led to a large number of theft and robbery crimes.
# MAGIC Through online research, the reason for the decline in crime rate since 2018 may be that the San Francisco Police Department has increased uniformed police patrols, hence violence and theft activities have been greatly reduced. In addition, the San Francisco Police Department stepped up its crackdown on the drug trade, which is also one of the reasons for the decline in crime rate.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime with respsect to the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

# MAGIC %md method1

# COMMAND ----------


# Show number of crime by hour for all records
q5_result = spark.sql("""
                      select substring(Time,1,2) as Hour,
                      count(*) as Count
                      from sf_crime
                      group by Hour
                      order by Hour
                      """)
display(q5_result)

# COMMAND ----------

# Show number of crime by hour for records in Christmas
q5_result = spark.sql("""
                      select substring(Time,1,2) as Hour,
                      count(*) as Count
                      from sf_crime
                      where Date like '12/25/%'
                      group by Hour
                      order by Hour
                      """)
display(q5_result)

# COMMAND ----------

# Show number of crime by hour for records in New Year Eve (12/31 - 01/01)
q5_result = spark.sql("""
                      select substring(Time,1,2) as Hour,
                             substring(Date,1,5) as Date_in_year,
                             count(*) as Count
                      from sf_crime
                      where Date like '12/31/%' or Date like '01/01/%'
                      group by Date_in_year, Hour
                      order by Date_in_year desc, Hour
                      """)
display(q5_result)

# COMMAND ----------

# MAGIC %md method2

# COMMAND ----------

from pyspark.sql.functions import to_timestamp
# to_timestamp function provided by pyspark
# purpose is to generate a new column with its defined logic

# sometimes functions provided by pyspark might not satify our usage
# then we want to define our own function, and apply the same way as functions provided by pyspark

# add new columns to convert Time to hour format
df_new1 = df_new.withColumn('IncidentTime', to_timestamp(df_new['Time'],'HH:mm')) 
# extract hour from incident time
df_new1 = df_new1.withColumn('Hour',hour(df_new1['IncidentTime']))
display(df_new1.take(5))

# COMMAND ----------

dates = ['12/15/2015','12/15/2016','12/15/2017']
df_days = df_new1[df_new1.Date.isin(dates)]
spark_df_q5_1 = df_days.groupby('Hour','Date').count().orderBy('Date','Hour')
display(spark_df_q5_1)

# COMMAND ----------

# MAGIC %md method3

# COMMAND ----------

##5 Define 1 udf
def time_to_hour(x):
  return x.split(':')[0]
# given a string HH:MM representing time, return a string representing hour of the given time

tth_udf = udf(lambda x: time_to_hour(x))
# tth_udf is now like to_timestamp, or hour

##5 groupby hour
df_opt4 = df_opt1.withColumn('Hour',tth_udf('Time')) # add a new column with hour extracted from time
df_hour_res = df_opt4.filter(df_opt4.Date =='12/15/2017').groupBy('Hour').count().orderBy('count')
display(df_hour_res)

# COMMAND ----------

##5 Visulize the result
df_hour_res_pd_df = df_hour_res.toPandas()
plt.figure()
ax = df_hour_res_pd_df.plot(kind ='bar',x='Hour',y='count',color = 'pink',legend = False)
ax.set_ylabel('count',fontsize = 12)
ax.set_xlabel('hour',fontsize = 12)
plt.xticks(fontsize=8, rotation=90)
plt.title('# 5 Total crimes in different hours on 12/15/2017')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 
# MAGIC

# COMMAND ----------

#df way
spark_df_q6_s1 = df_new.groupby('PdDistrict').count().orderBy('count',ascending = False)
display(spark_df_q6_s1)

# COMMAND ----------

top3_danger = df_new.groupby('PdDistrict').count().orderBy('count',ascending = False).head(3)
top3_danger_district = [top3_danger[i][0] for i in range(3)]
top3_danger_district

# COMMAND ----------

#sql way
spark_sql_q6_s1 = spark.sql( """
                             SELECT PdDistrict, COUNT(*) as Count
                             FROM sf_crime
                             GROUP BY 1
                             ORDER BY 2 DESC
                             LIMIT 3 
                             """ )
display(spark_sql_q6_s1)

# COMMAND ----------

# MAGIC %md Step2: Find out the crime event w.r.t category and time (hour) from the result of step 1

# COMMAND ----------

# df way
spark_df_q6_s2 = df_new1.filter(df_new1.PdDistrict.isin('SOUTHERN', 'MISSION', 'NORTHERN')).groupby('Category','Hour').count().orderBy('Category','Hour')
display(spark_df_q6_s2)
