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
parser.add_argument('--libbyURL', default = '', type = str,
                    help = 'url with json file of libby timeline')

# main script
if __name__ == "__main__":

    print('\n\n==========================================================')
    start = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('Starting at ' + str(start) + '\n\n') 

    # parse all arguments
    args = parser.parse_args()

    # read arguments
    libbyURL = args.libbyURL

    # get data from libby timeline 
    libbyList = getAllLibbyItems(libbyURL)
    print('Length of Libby list: ' + str(len(libbyList)))

    end = arrow.get(time.time()).to('US/Pacific').format('YYYY-MM-DD HH:mm:ss ZZ')
    print('\n\n' + 'Ending at ' + str(end)) 
    print('==========================================================\n\n')
