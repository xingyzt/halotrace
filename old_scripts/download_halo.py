#!/usr/bin/env python3
import illustris_python as il
import requests
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

halo_id = int(sys.argv[1])
repeat = False

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
    
snap_index = 99
snap_dir = f'snap_{snap_index:03d}/'

parent_request = { 'limit': '100' }
parent_url = f'{baseUrl}/snapshots/99/halos/{halo_id}/'
parent = get(parent_url,  params=parent_request )
parent_info = get(parent['meta']['info'])
halo_id = str(parent_info['InfoID'])
print(halo_id, parent_info)

parent_file = base_dir + snap_dir + f'cutout_{parent_info["InfoID"]}.hdf5'

if (repeat == False) and os.path.exists(parent_file):
    # with h5py.File("outputs/dm_proto.hdf5", "r") as f:
    with h5py.File(parent_file, 'r') as f:
        if 'PartType0/Coordinates' in f:
            print('halo already exists')
            sys.exit(0)

parent_request = {
    'gas':'Coordinates,Masses,ParticleIDs,Density,ElectronAbundance,StarFormationRate,GFM_Metals',
    'stars':'Coordinates,Masses,ParticleIDs,GFM_StellarFormationTime'
}
parent_cutout = get(parent_url + 'cutout.hdf5', parent_request, save_dir=snap_dir)

# sibling_subs = [ 
#     get(res['url']) for res in parent['child_subhalos']['results']
# ]
# # array entries
# sub_keys = set(sibling_subs[0].keys()) - { 'related', 'cutouts', 'trees', 'supplementary_data', 'vis', 'meta' }

# sibling_request = {
#     'gas':'ParticleIDs',
#     'stars':'ParticleIDs'
# }
# for sub in sibling_subs:
#     if sub['len_gas'] + sub['len_stars'] > 0:
#         get(sub['cutouts']['subhalo'], sibling_request, save_dir=snap_dir)

# with h5py.File(base_dir + snap_dir + f'cutout_{parent_info["InfoID"]}.hdf5', 'r') as f:
#     gas_ids = f['PartType0/ParticleIDs'][:]
#     star_ids = f['PartType4/ParticleIDs'][:]

#     gas_sub_ids = -np.ones_like(gas_ids, dtype=np.int64)
#     star_sub_ids = -np.ones_like(star_ids, dtype=np.int64)

# for sub in sibling_subs:
#     if sub['len_gas'] + sub['len_stars'] > 0:
#         with h5py.File(base_dir + snap_dir + f'cutout_{sub["id"]}.hdf5', 'r') as sub_f:
            
#             if sub['len_gas'] > 0:
#                 sub_gas_ids = sub_f['PartType0/ParticleIDs'][:]
#                 gas_sub_ids[np.isin(gas_ids, sub_gas_ids)] = sub["id"]
            
#             if sub['len_stars'] > 0:
#                 sub_star_ids = sub_f['PartType4/ParticleIDs'][:]
#                 star_sub_ids[np.isin(star_ids, sub_star_ids)] = sub["id"]

# with h5py.File(base_dir + snap_dir + f'cutout_{halo_id}.hdf5', 'a') as f:
#     f['PartType0/SubhaloIDs'] = gas_sub_ids
#     f['PartType4/SubhaloIDs'] = star_sub_ids

with h5py.File("outputs/dm_proto.hdf5", "a") as f:
    g_halo = f.require_group(f'halo_{halo_id}')
    for k, v in parent_info.items():
        if k in g_halo:
            del g_halo[k]
        g_halo[k] = v

#     g_subs = g_halo.create_group('subhalos')
#     for k in sub_keys:
#         g_subs[k] = [ s[k] for s in sibling_subs ]
