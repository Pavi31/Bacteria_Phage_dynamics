#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:21:03 2020

@author: karthik & Pavi
"""

import numpy as np

class Lattice:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def CreateLattice(self):
        XYGrid = np.zeros((self.height, self.width), dtype = int)
        return XYGrid