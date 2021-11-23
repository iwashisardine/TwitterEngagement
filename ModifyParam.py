import csv
import datetime
import locale
import tqdm
import pandas as pd

'''this is the only variable you need to change'''
datasetname = 'osusume_100'


import_path = f'./data/vec_data_{datasetname}.csv'
export_path = f'./data/Rpat2_{datasetname}.csv'

def day2num(day):
    dataset['Mon'] = (day == "Monday")*1.0
    dataset['Tue'] = (day == "Tuesday")*1.0
    dataset['Wed'] = (day == "Wednesday")*1.0
    dataset['Thu'] = (day == "Thursday")*1.0
    dataset['Fri'] = (day == "Friday")*1.0
    dataset['Sat'] = (day == "Saturday")*1.0
    dataset['Sun'] = (day == "Sunday")*1.0

def hour2num(hour):
    dataset['am0'] = (hour == 0)*1.0
    dataset['am1'] = (hour == 1)*1.0
    dataset['am2'] = (hour == 2)*1.0
    dataset['am3'] = (hour == 3)*1.0
    dataset['am4'] = (hour == 4)*1.0
    dataset['am5'] = (hour == 5)*1.0
    dataset['am6'] = (hour == 6)*1.0
    dataset['am7'] = (hour == 7)*1.0
    dataset['am8'] = (hour == 8)*1.0
    dataset['am9'] = (hour == 9)*1.0
    dataset['am10'] = (hour == 10)*1.0
    dataset['am11'] = (hour == 11)*1.0
    dataset['am12'] = (hour == 12)*1.0
    dataset['pm1'] = (hour == 13)*1.0
    dataset['pm2'] = (hour == 14)*1.0
    dataset['pm3'] = (hour == 15)*1.0
    dataset['pm4'] = (hour == 16)*1.0
    dataset['pm5'] = (hour == 17)*1.0
    dataset['pm6'] = (hour == 18)*1.0
    dataset['pm7'] = (hour == 19)*1.0
    dataset['pm8'] = (hour == 20)*1.0
    dataset['pm9'] = (hour == 21)*1.0
    dataset['pm10'] = (hour == 22)*1.0
    dataset['pm11'] = (hour == 23)*1.0

raw_dataset = pd.read_csv(import_path)
dataset = raw_dataset.copy()

time = dataset['time']
dt = datetime.datetime.strptime(time, '%Y-%m-%d_%H:%M:%S')
strday = dt.strftime('%A')
eng = int(dataset['liked']) + int(dataset['RT'])

day2num(strday)
hour2num(dt.hour)
dataset['engagement'] = eng

dataset.drop(columns=['time', 'liked', 'RT'])

try:
    dataset.drop(columns='get')
    dataset.drop(columns='id')
except:
    pass

dataset.to_csv(export_path)

'''produce pattern1 dataset'''
del_columns = ['follow', 'follower', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(13):
    del_columns.append(f'am{i}')
for i in range(11):
    del_columns.append(f'pm{i+1}')
export_path = f'./data/Rpat1_{datasetname}.csv'
dspat1 = dataset.copy()
dspat1.drop(columns=del_columns)
dspat1.to_csv(export_path)