#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 18:07:16 2023

Deploy xgboost long-term model (1982)
Run on savio

@author: yanghuikang
"""

import warnings
warnings.filterwarnings('ignore')

import time
import math
import sys
sys.path.insert(0, '/global/home/users/yanghuikang/projects/upscale/scripts/utility')
import model_deployment as md

# set temperary directory for data spill to scratch, otherwise by default data will spill to the home directory
import dask
dask.config.set({'temporary_directory': '/global/scratch/users/yanghuikang/upscale/data/temp'})
from dask.distributed import Client, LocalCluster

def main(argv):
    """
    Argvs:
        model_name: name of the model
        ens_version: ensemble model version (version is in directory name)
        output_version: output file version
        start_year: starting year to process
        ym: number of months after the starting year; 0 means Jan of the start year
    """
    
    # Dask local cluster
    cluster = LocalCluster(ip='*') 
    client = Client(cluster)
    print(client.dashboard_link, flush=True)
    
    model_name = argv[0]
    ens_version = argv[1]
    output_version = argv[2]
    start_year = int(argv[3])
    ym = int(argv[4])
    
    year = math.floor(ym/12) + start_year
    month = ym % 12 + 1

    # version = 'v1_2_0'
    
    start = time.time()
    md.model_deploy(model_name, ens_version, output_version, year, month, client)
    time_lapse = (time.time() - start)/60
    print('total time: {} minutes'.format(round(time_lapse, 1)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])