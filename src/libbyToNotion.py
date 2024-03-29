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

from lib.port_utils import portFullLibbyListToNotion
from lib.port_utils import updatePageCoversInNotion

from lib.utils import *

from globalStore import constants

# arguments
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--fullLibby', default = True, type = strToBool,
                    help = 'Should the full libby timeline be used or only the latest updates to it')
parser.add_argument('--oneTimeCoverUpdate', default = False, type = strToBool,
                    help = 'One time run through of the full libby timeline to update the cover images')
# main script
if __name__ == "__main__":

    print('\n\n==========================================================')
    start = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('Starting at ' + str(start) + '\n\n') 

    # parse all arguments
    args = parser.parse_args()

    # read arguments
    fullLibby = args.fullLibby
    oneTimeCoverUpdate = args.oneTimeCoverUpdate

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

    if oneTimeCoverUpdate:
        updatePageCoversInNotion(notion, notionDB_id, fileURL)

    end = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('\n\n' + 'Ending at ' + str(end)) 
    print('==========================================================\n\n')
