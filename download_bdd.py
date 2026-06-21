"""
BDD100K Dataset Downloader
==========================
Automatically downloads BDD100K dataset including images and segmentation masks.
Configures UTF-8 encoding for console output and downloads datasets to the 'datasets' folder.

Dependencies:
    - dataset_tools: Tool for downloading datasets
"""

import dataset_tools as dtools

import sys, io
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


dtools.download(dataset='BDD100K: Images 100K', dst_dir='datasets')
dtools.download(dataset='BDD100K: Segmentation', dst_dir='datasets')
