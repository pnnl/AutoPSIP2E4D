# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:53:10 2022

@author: john775 mang464

This code pushes completed PSIP data files from the local machine to the server. This code should be run on the local machine runing SIP data acquisition.

"""
import sys
import io
import os
import time
import datetime

#Define File Export Parameters
if len(sys.argv) < 2:
    print('Please specify the prefix of the data file name.')
    exit()
else:
    psip_pre = sys.argv[1]#'Dash'
    #dest_comp= 'mang464@tahoma.emsl.pnnl.gov'
    #dest_dir = '/tahoma/emsls60004/SIPfiles'
    #dest_comp = 'mang464@max.pnl.gov'
    #dest_dir = '/shared/mang464/AutoPSIP'
    dest_comp = 'mang464@spud.pnl.gov'
    dest_dir = '/home/mang464/AUTOSIP'
 
# Identify local directories for read and write
local_dir = 'C:\\nginx-1.7.3\\PSIP\\logs'
local_copy = 'C:\\nginx-1.7.3\\PSIP\\logs\\ToHPC'

# Get some information about the length of strings and confirm variables
npre = len(psip_pre)
ncp = len(dest_comp)
ndr = len(dest_dir)

# Move to local directory
os.system('cd '+local_dir)

#loop to check for new PSIP data files
while True:
    print (time.strftime("%I:%M:%S")),'checking for new data files'
    for fil in os.listdir(local_dir):
        if (psip_pre == fil[:npre]) & fil.endswith('.csv'):
            print('Found a matching file, evaluating...')

            #make sure the file is done, sleep for 1 minutes to see if file size changes
            done = False
            fsize1 = os.path.getsize(local_dir+'\\'+fil);
            time.sleep(60)
            fsize2 = os.path.getsize(local_dir+'\\'+fil);
            if fsize1 == fsize2:
                #send this file to the remote computer
                str1 = 'pscp "'+local_dir+'\\'+fil+'" '+dest_comp+':'+dest_dir
                os.system(str1)
                print fil,' has been exported'
                #move this file to local directory of copied files
                os.system('move '+local_dir+'\\'+fil+' '+local_copy)
            if fsize1!=fsize2:
                print(fil,' is incomplete')
        time.sleep(0.01)
