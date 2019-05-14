
import pandas as pd

f = open("SMSSpamCollection.txt", "r+")

#create dataframe
attr = ['type','text']
data_frame = pd.DataFrame(columns=attr)

f1=f.readlines()
count = 0
for x in f1:
    type = x.split("\t")
    data_frame.loc[count] = type
    count = count +1

print(data_frame.head(5))

