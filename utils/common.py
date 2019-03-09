#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
some path related util.
'''
import sys, os, configparser

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
CONFIG_FILE = os.path.join(SCRIPT_PATH, 'config.ini')

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_config(config=CONFIG_FILE):
    cf = configparser.ConfigParser()
    cf.read(config)
    return cf


