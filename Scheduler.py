import requests
from datetime import datetime
data = requests.get("http://worldtimeapi.org/api/timezone/Asia/Kolkata")
date = data.json()['datetime'].split('T')[0]
date = date[2:]
date = datetime.strptime(date, '%y-%m-%d')

# This will check if its more then 19 days for our model to get retrained
def isscheduled():
    with open('prevdays.txt') as file:
        previous_date = file.readline()
        previous_date = previous_date.strip()
        file.close()
    previous_date = datetime.strptime(previous_date[2:], '%y-%m-%d')
    if str(date-previous_date).split(' ')[0] == '10':
        return True 
    else:
        return False