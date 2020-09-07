#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
# Copyright (c) 2020 Christa Cuchiero, Wahid Khosrawi, Josef Teichmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""helpers.py:
This file implements some helper functionality. 
"""

import numpy as np, scipy as sp, tensorflow as tf
import os, time

class reporter():
    def __init__(self):
        self.logMaster = 'log.txt'
        self.rmMaster  = 'README.md'
        self.caliFolder = 'caliRes'

        # Make sure the dest folder exists
        if not os.path.exists(self.caliFolder):
            os.makedirs(self.caliFolder)
        

    def writeLog(self,message):
        if type(message) != list:
            raise TypeError('WriteLog accepts only lists for writing')
        
        with open(self.logMaster, 'a') as f:
            for line in message:
                f.writelines(line+'\n')




def init_folders_and_readme(*, mode, version):
    
    if mode == 'create_data':
        if not os.path.exists('caliRes/'+version):
            os.makedirs('caliRes/'+version)

            lines = ["Version: "+version+"\n\n",
                    "Plotting implied vola from local vol model\n\n",
                    "We use the model parameters:\n",
                    "We use the loc vol parameters:\n\n"
            ]
        
            with open('caliRes/'+version+'/README.md','w') as file:
                file.writelines(lines)

            lines = ['Log file with std output\n\n',
                     '------------------------------------']
            with open('caliRes/'+version+'/log.txt','w') as file:
                file.writelines(lines)



    elif mode == 'calibrate SABR':
        if not os.path.exists('caliRes/'+version):
            ans = input('This folder should exist and contain the data plots, should I proceed? (y/n)')
            if ans == "y":
                print("OK, I continue")
                os.makedirs('caliRes/'+version)
            else:
                raise("I was told to quit since the folder should exist")

            

    elif mode == 'calibrate':
        if not os.path.exists('caliRes/'+version):
            os.makedirs('caliRes/'+version)

            lines = ["Version: "+version+"\n\n",
                    "Calibrating implied vola from data: {}\n\n".format(version),
                    #"We use the model parameters:\n",
                    #"alpha0 : {}\n\n\n".format(para['alpha0']),
            ]
        
            with open('caliRes/'+version+'/README.md','w') as file:
                file.writelines(lines)

            lines = ['Log file with std output\n\n',
                     '------------------------------------']
            with open('caliRes/'+version+'/log.txt','w') as file:
                file.writelines(lines)

        elif version != 'AUX':
            answer = input('This folder already exists, do you want me to overwrite?')
            if answer in ['y','Y','yes','Yes']:
                os.popen('rm -rf caliRes/'+version)
                print('Removed the folder')
                time.sleep(2)
                init_folders_and_readme(mode=mode, version=version)
            else:
                raise Exception('This version folder for the calibration already exists.')

    else:
        raise Exception('Unknown mode here, check!')
        
        

if __name__ == "__main__":
    pass