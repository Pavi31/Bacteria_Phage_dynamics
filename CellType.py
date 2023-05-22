#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:21:52 2020

@author: karthik & Pavi
"""

import random as rand
import numpy as np 


class CellType:
    def __init__(self, name, ID, K, r, N0):
        self.name = name
        self.ID = ID
        self.K = K
        self.r = r
        self.N0 = N0
        self.Nt = 0
        self.NtMinus = 0
    