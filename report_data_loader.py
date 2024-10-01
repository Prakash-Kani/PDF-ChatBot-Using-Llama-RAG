from pymongo import MongoClient
import os
from dotenv import load_dotenv


def data_loader(course_name):
    # Load environment variables from .env file
    load_dotenv()
    connection_link = os.getenv("CONNECTION_LINK")
    collection = MongoClient(connection_link).imaginx.chats
    lst = []
    if course_name.lower() == 'all' or 'all' in course_name.lower():
        datas = collection.find({}, {'_id':0})
    else:
        datas = collection.find({'filename':course_name}, {'_id':0})
    
    
    for data in datas:
        lst.append(data)
    import pandas as pd
    df = pd.DataFrame(lst)
    df[df.columns[0:4]].to_csv(r'Reports/reports_data.csv')
    print('Report Data Loaded!')