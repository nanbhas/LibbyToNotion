"""
Author: Nandita Bhaskhar
Test script for reading Notion API correctly
"""

import sys
sys.path.append("../")

import arrow
import time
import json

import globalStore.constants as constants

from notion_client import Client
from lib.port_utils import getAllRowsFromNotionDatabase


# main script
if __name__ == "__main__":

    print('\n\n==========================================================')
    start = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('Starting at ' + str(start) + '\n\n') 


    # open secrets
    with open(constants.NOTION_SECRET_FILE, "r") as f:
        secrets_notion = json.load(f)

    # initialize notion client and determine notion DB
    notion = Client(auth = secrets_notion['notionToken'])
    notionDB_id = secrets_notion['databaseID']

    # read data from notion
    print('Starting reading data in notion')
    allNotionRows = getAllRowsFromNotionDatabase(notion, notionDB_id)

    end = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('\n\n' + 'Ending at ' + str(end)) 
    print('==========================================================\n\n')