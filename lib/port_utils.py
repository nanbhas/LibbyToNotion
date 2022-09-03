"""
Author: Nandita Bhaskhar
Notion helper functions
"""

from select import select
import sys
sys.path.append('../')

import pprint
import time
import arrow
import json
import urllib.request

from globalStore import constants

from tqdm import tqdm

def getAllRowsFromNotionDatabase(notion, notionDB_id):
    '''
    Gets all rows (pages) from a notion database using a notion client
    Args:
        notion: (notion Client) Notion client object
        notionDB_id: (str) string code id for the relevant database
    Returns:
        allNotionRows: (list of notion rows)

    '''
    start = time.time()
    hasMore = True
    allNotionRows = []
    i = 0

    while hasMore:
        if i == 0:
            try:
                query = notion.databases.query(
                                **{
                                    "database_id": notionDB_id,
                                    #"filter": {"property": "UID", "text": {"equals": it.id}},
                                }
                            )
            except:
                print('Sleeping to avoid rate limit')
                time.sleep(30)
                query = notion.databases.query(
                                **{
                                    "database_id": notionDB_id,
                                    #"filter": {"property": "UID", "text": {"equals": it.id}},
                                }
                            )
                
        else:
            try:
                query = notion.databases.query(
                                **{
                                    "database_id": notionDB_id,
                                    "start_cursor": nextCursor,
                                    #"filter": {"property": "UID", "text": {"equals": it.id}},
                                }
                            )
            except:
                print('Sleeping to avoid rate limit')
                time.sleep(30)
                query = notion.databases.query(
                                **{
                                    "database_id": notionDB_id,
                                    "start_cursor": nextCursor,
                                    #"filter": {"property": "UID", "text": {"equals": it.id}},
                                }
                            )
            
        allNotionRows = allNotionRows + query['results']
        nextCursor = query['next_cursor']
        hasMore = query['has_more']
        i+=1

    end = time.time()
    print('Number of rows in notion currently: ' + str(len(allNotionRows)))
    print('Total time taken: ' + str(end-start))

    return allNotionRows


def getAllLibbyItems(fileURL, onlyBorrowed = True):
    '''
    Gets a list of all unique items (borrowed books) in Libby timeline with each item represented by a dict
    Args: 
        fileURL: (str) the json file url with Libby timeline export data or path to local json file
        onlyBorrowed: (bool) whether to only include borrowed items or all items
    Returns:
        libbyList: (list of dicts) each dict is a borrowed book with relevant keys like Title, Author, etc
    '''

    if fileURL.startswith('http'):
        with urllib.request.urlopen(fileURL) as url:
            data = json.loads(url.read().decode())
    else:
        with open(fileURL) as f:
            data = json.load(f)

    ibsnList = []
    libbyList = []
    for item in tqdm(data['timeline']):
        if onlyBorrowed:
            if item['activity'] == 'Borrowed':
                if item['isbn'] not in ibsnList:
                    prop = {}
                    prop['Name'] = item['title']['text']
                    prop['Author'] = item['author']
                    prop['Format'] = item['cover']['format']
                    prop['LibbyDate'] = str(arrow.get(item['timestamp']).to('US/Pacific').date())
                    prop['ISBN'] = item['isbn']
                    ibsnList.append(item['isbn'])
                    prop['Status'] = 'libby-inbox'
                    prop['CoverURL'] = item['cover']['url']
                    libbyList.append(prop)
        else:
            if item['isbn'] not in ibsnList:
                prop = {}
                prop['Name'] = item['title']['text']
                prop['Author'] = item['author']
                prop['Format'] = item['cover']['format']
                prop['LibbyDate'] = str(arrow.get(item['timestamp']).to('US/Pacific').date())
                prop['ISBN'] = item['isbn']
                ibsnList.append(item['isbn'])
                prop['Status'] = 'libby-inbox'
                prop['CoverURL'] = item['cover']['url']
                libbyList.append(prop)

    return libbyList

def getNotionPageCoverImageFromProp(prop):
    '''
    Get the cover image url from the prop dict
    Args:
        prop: (dict)
    Returns:
        notionCover: (dict)
    '''

    notionCover = {
    'type': 'external', 
    'external': {
        'url': prop['CoverURL']
        }
    }
    return notionCover


def getNotionPageEntryFromProp(prop):
    '''
    Format the prop dict to Notion page format
    Args:
        prop: (dict)
    Returns:
        newPage: (Notion formated dict)
    '''
    
    dateKeys = {'LibbyDate'}
    titleKeys = {'Name'}
    selectKeys = {'Status', 'Format'}
    coverKeys = {'CoverURL'}
    textKeys = set(prop.keys()) - dateKeys - titleKeys - selectKeys - coverKeys
    newPage = {
        "Name": {"title": [{"text": {"content": prop['Name']}}]}
    }
    
    for key in textKeys:
        newPage[key] = {"rich_text": [{"text": { "content": prop[key]}}]}    
    
    for key in dateKeys:
        newPage[key] = {"date": {"start": prop[key] }}
        
    for key in selectKeys:
        newPage[key] = {'select': {'name': prop[key]} }
    
    return newPage


def portFullLibbyListToNotion(notion, notionDB_id, fileURL = None):
    '''
    Port all libby items (unique) to Notion ensuring that there are no duplicates
    Args:
        notion: (notion Client) Notion client object
        notionDB_id: (str) string code id for the relevant database
        fileURL: (default None, else str representing an url) if provided should be the json file url with Libby timeline export data
    '''

    if not fileURL:
        with open(constants.LIBBY_SECRET_FILE, "r") as f:
            secrets_libby = json.load(f)
        fileURL = secrets_libby["url"]

    allNotionRows = getAllRowsFromNotionDatabase(notion, notionDB_id)

    libbyList = getAllLibbyItems(fileURL)

    for prop in tqdm(libbyList):

        # check if it is already in notion
        try:
            notionRow = [row for row in allNotionRows if row['properties']['Name']['title'][0]['text']['content'] == prop['Name']]
        except:
            # notion page does not have a title
            notionRow = []

        if len(notionRow) >= 1: # duplicate already present

            notionRow = notionRow[0]

            if notionRow['cover']: # cover already present
                print('Skipping. Already present: ', prop['Name'])
                pass
            else: # cover not present
                print('Updating cover: ', prop['Name'])
                pageID = notionRow['id']
                notionCover = getNotionPageCoverImageFromProp(prop)
                notion.pages.update(pageID, cover = notionCover)

        else: # entry not present in notion

            notionPage = getNotionPageEntryFromProp(prop)
            notionCover = getNotionPageCoverImageFromProp(prop)
            try:
                print('Creating entry for: ', prop['Name'])
                notion.pages.create(parent={"database_id": notionDB_id}, cover = notionCover, properties = notionPage)
            except:
                print('Sleeping to avoid rate limit')
                time.sleep(30)
                notion.pages.create(parent={"database_id": notionDB_id}, cover = notionCover, properties = notionPage)


def updatePageCoversInNotion(notion, notionDB_id, fileURL = None):
    '''
    Update all page covers of all libby items in Notion
    Args:
        notion: (notion Client) Notion client object
        notionDB_id: (str) string code id for the relevant database
        fileURL: (default None, else str representing an url) if provided should be the json file url with Libby timeline export data
    '''

    if not fileURL:
        with open(constants.LIBBY_SECRET_FILE, "r") as f:
            secrets_libby = json.load(f)
        fileURL = secrets_libby["url"]

    allNotionRows = getAllRowsFromNotionDatabase(notion, notionDB_id)

    libbyList = getAllLibbyItems(fileURL, onlyBorrowed=False)

    libbyTitles = [ item['Name'] for item in libbyList ]
    pprint.pprint(libbyTitles)
    print('\n\n')

    for notionRow in allNotionRows:

        if notionRow['properties']['Status']['select']['name'] == 'libby-inbox':

                try:
                    prop = [prop for prop in libbyList if prop['ISBN'] == notionRow['properties']['ISBN']['rich_text'][0]['text']['content']][0]
                    print('Found libby item: ', prop['Name'])
                except:
                    # libby item not found
                    prop = None
    
                if prop:
                    if notionRow['cover']:
                        print('Skipping. Already has cover: ', prop['Name'])
                    else:
                        print('Updating cover: ', prop['Name'])
                        pageID = notionRow['id']
                        notionCover = getNotionPageCoverImageFromProp(prop)
                        notion.pages.update(pageID, cover = notionCover)