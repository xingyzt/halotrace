#!/usr/bin/env python3
import illustris_python as il
import requests
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sub_index = int(sys.argv[1])

baseUrl = 'http://www.tng-project.org/api/TNG-Cluster'
headers = {"api-key":os.getenv('API_KEY')}
base_dir = 'tng_cache/'

def get(path, params=None, save_dir=''):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = base_dir + save_dir + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

sim = get(baseUrl)

# get z = 0 snapshot
snap_index = 99
snap_dir = f'snap_{snap_index:03d}/'

snaps = get(sim['snapshots'])
snap = get(snaps[snap_index]['url'], save_dir=snap_dir)

subs = get( snap['subhalos'], {'limit':50, 'order_by':'-mass_stars'})

# get (sub_index)th most massive subhalo and its siblings
sub = get( subs['results'][sub_index]['url'] )

parent_request = { 'limit': '100' }
parent = get(sub['related']['parent_halo'], params=parent_request )
parent_info = get(parent['meta']['info'])
print(parent_info['InfoID'], parent_info)

with h5py.File("outputs/dm_proto.hdf5", "a") as f:
    key = f'halo_{parent_info["InfoID"]}'
    if key in f.keys():
        g_halo = f[key]
    else:
        g_halo = f.create_group(key)
    for i in parent_info:
        if i in g_halo.keys():
            del g_halo[i]
        g_halo[i] = parent_info[i]
