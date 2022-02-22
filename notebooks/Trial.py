# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import json
import random
import string
import time
from datetime import datetime
import arrow

# %%
import numpy as np
import pandas as pd

# %%
import os
import sys
sys.path.append('../')

# %%
from notion_client import Client

# %%
from globalStore import constants

# %%
from lib.port_utils import *

# %%
with open("../secrets/secrets_notion.json", "r") as f:
    secrets_notion = json.load(f)

# %%
with open("../secrets/secrets_libby.json", "r") as f:
    secrets_libby = json.load(f)

# %%
notion = Client(auth = secrets_notion['notionToken'])

# %%
notionDB_id_A = secrets_notion['databaseID_A']
notionDB_id_B = secrets_notion['databaseID_B']

# %%

# %%

# %% [markdown]
# ## Trial with libby data function

# %%
fileURL = secrets_libby["url"]
ld = getAllLibbyItems(fileURL)

# %%
len(ld)

# %%
pd.DataFrame(ld)

# %%
getNotionPageEntryFromProp(ld[0])

# %% tags=[]
portFullLibbyListToNotion(notion, notionDB_id_A)

# %%
allNotionRows = getAllRowsFromNotionDatabase(notion, notionDB_id_A)

# %%
notionRow = [row for row in allNotionRows if row['properties']['Name']['title'][0]['text']['content'] == ld[0]['Name']]

# %%
len(notionRow)

# %%

# %% [markdown]
# ## Trials with libby data

# %%
import urllib.request, json 
fileURL = secrets_libby["url"]
with urllib.request.urlopen(fileURL) as url:
    data = json.loads(url.read().decode())
    #print(data)

# %%
tt = data['timeline']

# %%
bb = [item for item in tt if item['activity'] == 'Borrowed']

# %%
len(bb)

# %%
data['timeline'][0].keys()

# %%
data['timeline'][0]['cover']

# %%
data['timeline'][0]['title']

# %%
data['timeline'][0]['author']

# %%
data['timeline'][0]['isbn']

# %%
arrow.get(data['timeline'][0]['timestamp'])#.to('US/Pacific')

# %%
arrow.get(data['timeline'][0]['timestamp']).to('US/Pacific')

# %%
arrow.get(data['timeline'][0]['timestamp']).to('US/Pacific').date()

# %%
data['timeline'][0]["activity"]

# %%
data['timeline'][0]["details"]

# %% jupyter={"outputs_hidden": true} tags=[]
for item in data['timeline']:
    if item['activity'] == 'Borrowed':
        print(item['title']['text'])
        print(item['author'])
        print(item['cover']['format'])
        print("Image: ", item['cover']['url'])
        print(item['activity'])
        print(arrow.get(item['timestamp']).to('US/Pacific').date())
        print(item['isbn'])
        print('----------')

# %% [markdown]
# ## See how to set select keys

# %%
allRows = getAllRowsFromNotionDatabase(notion, notionDB_id_A)

# %%
allRows[0]

# %%
allRows[-1]

# %%
pageID = allRows[-2]['id']

# %%
allRows[-1]['properties']['Status']

# %%
notion.pages.update( pageID, properties = {'Status': {'select': {'name': 'Not started'} } })

# %%
{'id': '1', 'name': 'Not started', 'color': 'red'}

# %%

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## Query pages from Database whose icons are unset

# %%
allRows = getAllRowsFromNotionDatabase(notion, notionDB_id_A)

# %%
for x in np.arange(11):
    if allRows[x]['icon'] is None:
        print(x)

# %%
allRows[1]

# %%
todayMorning = arrow

# %%
arrow

# %%

# %%

# %%

# %%
notionDB_id = notionDB_id_A
start = time.time()
hasMore = True
allNotionRows = []
i = 0

while hasMore:
    if i == 0:
        # try:
            query = notion.databases.query(
                            **{
                                "database_id": notionDB_id,
                                "filter": {"property": "created_time", "after": ""},
                            }
                        )
        # except:
        #     print('Sleeping to avoid rate limit')
        #     time.sleep(30)
        #     query = notion.databases.query(
        #                     **{
        #                         "database_id": notionDB_id,
        #                         "filter": {"icon": None},
        #                     }
        #                 )

    else:
        # try:
            query = notion.databases.query(
                            **{
                                "database_id": notionDB_id,
                                "start_cursor": nextCursor,
                                "filter": {"icon": None},
                            }
                        )
        # except:
        #     print('Sleeping to avoid rate limit')
        #     time.sleep(30)
        #     query = notion.databases.query(
        #                     **{
        #                         "database_id": notionDB_id,
        #                         "start_cursor": nextCursor,
        #                         "filter": {"icon": None},
        #                     }
        #                 )

    allNotionRows = allNotionRows + query['results']
    nextCursor = query['next_cursor']
    hasMore = query['has_more']
    i+=1

end = time.time()
print('Number of rows in notion currently: ' + str(len(allNotionRows)))
print('Total time taken: ' + str(end-start))

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Create a new page with Date fields and set icon

# %%
# - Database: Trial A with prop Date
# - Create a page, title it as (today’s) mm/dd and set date to today. Set icon to a fixed icon

# %%
DT = arrow.now().to('US/Pacific').date()
DT

# %%
arrow.get(DT)

# %%
str(arrow.get(DT))

# %%
datetime.now().strftime("%m/%d")

# %%
datetime.today().strftime("%m/%d")

# %%
datetime.today()

# %%
datetime.now()

# %%
title = DT.strftime("%m/%d")

# %%
dateIso = datetime.today().isoformat() #arrow.now().to('US/Pacific').isoformat()
dateIso

# %%
allRows = getAllRowsFromNotionDatabase(notion, notionDB_id_A)

# %%
allRows[0]

# %%
newPage = {}
newPage["Name"] = {"title": [{"text": {"content": title }}]}
newPage["Date"] = {"date": {"start": dateIso } }
iconUrl = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/02337281-46ff-4fe8-8e26-378ed8bfbf1f/basketball.svg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220201T031439Z&X-Amz-Expires=3600&X-Amz-Signature=ee44ae7988ff20d0ecfc93a53329136e8444528d290f5d4d3d4c863e481a749c&X-Amz-SignedHeaders=host&x-id=GetObject"
# ic = {'icon': {'type': 'file',
#   'file': {'url': 'https://s3.us-west-2.amazonaws.com/secure.notion-static.com/02337281-46ff-4fe8-8e26-378ed8bfbf1f/basketball.svg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220201T031439Z&X-Amz-Expires=3600&X-Amz-Signature=ee44ae7988ff20d0ecfc93a53329136e8444528d290f5d4d3d4c863e481a749c&X-Amz-SignedHeaders=host&x-id=GetObject',
#    'expiry_time': '2022-02-01T04:14:39.300Z'}}}
ic = { "external": {"url":"https://img.icons8.com/material/280/BDC3C8/circled-a.png"}} 
# ic = {"file": {"url": "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/02337281-46ff-4fe8-8e26-378ed8bfbf1f/basketball.svg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220201T031439Z&X-Amz-Expires=3600&X-Amz-Signature=ee44ae7988ff20d0ecfc93a53329136e8444528d290f5d4d3d4c863e481a749c&X-Amz-SignedHeaders=host&x-id=GetObject",
#    "expiry_time": "2022-02-01T04:14:39.300Z"} }

# %%
notion.pages.create(parent={"database_id": notionDB_id_A}, icon=ic, properties = newPage)

# %% [markdown]
# ## Update icons of pages queried from Database

# %%
allRows = getAllRowsFromNotionDatabase(notion, notionDB_id_A)

# %%
for row in allRows:
    pageID = row['id']
    try:
        notion.pages.update( pageID, icon = ic)
    except:
        print('Sleeping to avoid rate limit')
        time.sleep(30)
        notion.pages.update( pageID, icon = ic)


# %%
allRows[0]['id']

# %% [markdown]
# ## Twitter trial changes

# %%
allRows = getAllRowsFromNotionDatabase(notion, notionDB_id_A)

# %%
allRows[4]

# %%
notion.blocks.children.list(allRows[4]['id'])['results']

# %%
