import pandas as pd
import datetime


start = datetime.datetime.today()
print(start)
df = pd.read_csv('./mimiciii/CHARTEVENTS.csv')
end = datetime.datetime.today()

print("DONE")
print(start - end)