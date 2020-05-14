#!/usr/bin/env python

#  Author:  Ben Tannenwald
##  Date:    May 14, 2020
##  Purpose: Script to download /drug/event case study files from openFDA

import pandas as pd
import numpy as np
import os

# ***0. Get json from openFDA that contains links to all files
os.system("wget https://api.fda.gov/download.json")
drugEventJSON = pd.read_json('download.json')['results']['drug']['event']

# needed structure-> results/drug/event

print("Drug/Event keys: {}".format(drugEventJSON.keys()))

subDirs = []
totalRecords = drugEventJSON['total_records']
processedRecords = 0
totalSize = 0

print(drugEventJSON['partitions'][0])
exit

for partition in drugEventJSON['partitions']:
    subDir = partition['display_name'].split(' (')[0].replace(' ', '-')
    json   = partition['file'].split(' (')[0].replace(' ', '-')
    filename = partition['file'].split('/')[-1]
    
    totalSize = float(partition['size_mb'])
    processedRecords += int(partition['records'])

    if subDir not in subDirs:
        os.mkdir(subDir)
        subDirs.append( subDir )

    os.system('wget -O {}/{} {}'.format( subDir, filename, partition['file']) )
    

downloadedFiles = 0
for subdir, dirs, files in os.walk('/home/btannenw/Desktop/life/adverseDrugFDA/data/'):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith('.json.zip') or filepath.endswith('.json.zip'):
            #print (filepath)
            downloadedFiles += 1
            
print('Number of subDirs: {}'.format( len(subDirs) ))
print('Total Records: {}'.format( totalRecords))
print('Processed Records: {}'.format( processedRecords))
print('Total size [MB]: {}'.format( totalSize))
print('Downloaded Files: {}'.format( downloadedFiles))
