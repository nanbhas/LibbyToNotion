"""
Author: Nandita Bhaskhar
End-to-end script for getting updates from Libby on to the Notion Database of your choice
"""

import sys
sys.path.append('../')

import time
import arrow
import json

from notion_client import Client

from lib.port_utils import getAllRowsFromNotionDatabase, getAllLibbyItems
from lib.port_utils import getNotionPageEntryFromProp
from lib.port_utils import portFullLibbyListToNotion

from lib.utils import *

from globalStore import constants

# arguments
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--fullLibby', default = True, type = strToBool,
                    help = 'Should the full libby timeline be used or only the latest updates to it')

# main script
if __name__ == "__main__":

    print('\n\n==========================================================')
    start = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('Starting at ' + str(start) + '\n\n') 

    # parse all arguments
    args = parser.parse_args()

    # read arguments
    fullLibby = args.fullLibby

    # open secrets 
    with open(constants.LIBBY_SECRET_FILE, "r") as f:
        secrets_libby = json.load(f)
    with open(constants.NOTION_SECRET_FILE, "r") as f:
        secrets_notion = json.load(f)

    # get libby timeline file url
    fileURL = secrets_libby["url"]

    # initialize notion client and determine notion DB
    notion = Client(auth = secrets_notion['notionToken'])
    notionDB_id = secrets_notion['databaseID']

    if fullLibby:
        portFullLibbyListToNotion(notion, notionDB_id, fileURL)
    else:
        NotImplementedError
