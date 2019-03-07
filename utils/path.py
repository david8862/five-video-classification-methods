#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
some path related util.
'''
import os

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

