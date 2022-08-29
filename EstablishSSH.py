# -*- coding: utf-8 -*-
"""
Created on January 18, 2022

@author: mang464

This establishes and SSH connection to a server using authorized keys. This should be run on the local computer running the PSIP acquisiton.

"""
import os

putty = '"C:\\Program Files\\PuTTy\\putty.exe"'
#host = 'tahoma.emsl.pnl.gov';
#host = 'max.pnl.gov'
host = 'spud.pnl.gov'
user = 'mang464'
ctype = '-ssh'
os.system(putty+' '+ctype+' '+user+'@'+host)

#PSIP1329!
