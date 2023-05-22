#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:25:19 2020

@author: karthik & Pavi
"""


from CellType import CellType
import random as rand
import numpy as np
from tabulate import tabulate
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import time
from functools import wraps

    
class Simulate():
    def __init__(self, T, dT, XYMatrix, log = False):
        self.T = T
        self.dT = dT
        self.XYMatrix = XYMatrix
        self.XYFrames = []
        self.LtoPhageRate = 0.
        self.PhagesPerLDeath = 100
        self.StoLRate = 0.
        self.LtoLgRate = 0.
        self.rS = 0.
        self.StoDeathRate = 0.
        self.diffusivity = 0.
        self.MeanVariation = 0.
        self.K = 70000
        self.SCell = CellType("S Cell", 1, self.K, 0, 0)
        self.Phage = CellType("Phages", 2, self.K, 0, 0)
        self.LCell = CellType("L Cell", 3, self.K, 0, 0)
        self.LgCell = CellType("Lg Cell", 4, self.K, 0, 0)
 
        self.PhageKills = 0 # USed for logging
        self.LysogensFormed = 0 # Used for logging
        self.StoLConversions = 0
        self.Sdeaths = 0
        self.log = log
        self.StoDeathConversions = 0.
        self.tlist = [] #Used for logging
        self.SCelllist = []
        self.Phageslist = []
        self.LCelllist = []
        self.LgCelllist = []
      
    
# def fn_timer(function):
#     @wraps(function)
#     def function_timer():
#         t0=time.time()
#         result = function()
#         t1=time.time()
#         print("Total time running %s: %s seconds" %(function.func_name, str(t1-t0)))
#         return result
#     return function_timer()

    def GenerateNo(self, Matrix, N, ID):
        n = 0
        while(n < N):
            Randomspot = np.random.choice(np.arange(int(0.1*len(Matrix)), len(Matrix)-int(0.1*len(Matrix))),2)
            if(Matrix[Randomspot[0], Randomspot[1]] == 0):
                Matrix[Randomspot[0], Randomspot[1]] = ID
                n+=1
            else:
                continue
        return Matrix
    
    def LogisticGrowth(self, NoOfCells, NoOfSCells, NoOfLCells, NoOfLgCells, r, K):
        N = NoOfLgCells + NoOfSCells + NoOfLCells
        return NoOfCells + NoOfCells*r*(1 - N/K)
    
    
    def Calculate_dN(self, Nt, NtMinus):
        return Nt - NtMinus
    

    def Transmutation(self, NoOfTransmutation, Matrix, DecayID, ID):
        NoOfConversions = NoOfTransmutation
        X,Y = np.where(Matrix==ID)
        if(len(X) > 0 and NoOfConversions > 0):
            index = np.random.choice(np.arange(0,len(X)), NoOfConversions)
            for i in index:
                Matrix[X[i]][Y[i]] = DecayID
        return Matrix



    def CheckNeighbours(self, Matrix, SeedX, SeedY, ID):
        choiceList = [(1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]
        neighbourCoord = [-99,-99]
        NotFound = True
        while(NotFound and len(choiceList) > 0):
            randomNeighbourList = rand.choices(choiceList, k=1)
            randomNeighbour = randomNeighbourList[0]
            if(SeedX+randomNeighbour[0] >= len(Matrix) or SeedX+randomNeighbour[0] < 0 or SeedY+randomNeighbour[1] >= len(Matrix[0]) or SeedY+randomNeighbour[1] < 0):
                break
            elif(Matrix[SeedX+randomNeighbour[0],SeedY+randomNeighbour[1]] == 0):
                neighbourCoord = [SeedX+randomNeighbour[0], SeedY+randomNeighbour[1]]
                NotFound = False 
            elif(Matrix[SeedX+randomNeighbour[0],SeedY+randomNeighbour[1]] < ID and Matrix[SeedX+randomNeighbour[0],SeedY+randomNeighbour[1]] > 0):
                neighbourCoord = [SeedX+randomNeighbour[0], SeedY+randomNeighbour[1]]
                
                NotFound = False
                n_id = Matrix[SeedX+randomNeighbour[0],SeedY+randomNeighbour[1]]
                # print(n_id, neighbourCoord, self.tlist[-1]+1)
                Matrix = self.PutPoints(Matrix, 1, n_id )
                # print(n_id, neighbourCoord, self.tlist[-1]+1)
            else:
                choiceList.remove(randomNeighbour)

        return neighbourCoord, Matrix


    def DelElements(self, Matrix, ID, dN):
        X,Y = np.where(Matrix==ID)
        if(len(X) > 0):
            index = np.random.choice(np.arange(len(X)), dN)
            Matrix[X[index], Y[index]] = 0
        return Matrix    
    
    def PutPoints(self, Matrix, dN, ID):
        n = 0
        X,Y = np.where(Matrix == ID)
        X = list(X)
        Y = list(Y)
        while(n < dN and len(X) > 0):
            index = rand.randint(0, len(X)-1)
            SeedX = X[index]
            SeedY = Y[index]
            neighbourCoord, Matrix = self.CheckNeighbours(Matrix, SeedX, SeedY, ID)
            if(neighbourCoord[0] < 0):
                #print("Point Stuck in between all Cells. Calculating new centers for splitting ")
                X.pop(index)
                Y.pop(index)
                if(len(X)==0):
                    print("All the points of ID : {} has no free sites to split dN = {}; n = {}".format(ID, dN, n))

            else:
                Matrix[neighbourCoord[0], neighbourCoord[1]] = ID
                n+=1
                X.append(neighbourCoord[0])
                Y.append(neighbourCoord[1])
    
        return Matrix

    
        
    
    def GhostFunc(self, a, size):
        if(a<0):
            return 0
        elif(a>=size):
            return size - 1
        else:
            return a
    
    def CoordinateReturn(self, size, x, y, diffusivity):
        leftend = self.GhostFunc(x-diffusivity, size)
        rightend = self.GhostFunc(x+diffusivity, size)
        bottom = self.GhostFunc(y-diffusivity, size)
        top = self.GhostFunc(y+diffusivity, size)
        return leftend, rightend, bottom, top
    
    
    # def KillSCells(self, Matrix, X, Y, ID, phageID):
    #     diffusivity = self.diffusivity       
    #     for x, y in zip(X,Y):
    #         # Find all S cells in the given radius
    #         leftend, rightend, bottom, top = self.CoordinateReturn(len(Matrix), x, y, diffusivity)
    #         Sindex_X, SindexY = np.where(Matrix[leftend:rightend+1,bottom:top+1]==ID)
            
    #         for i,j in zip(leftend+Sindex_X, bottom+SindexY):
    #             if(rand.random() <= abs(rand.gauss(self.StoDeathRate, self.StoDeathRate*self.MeanVariation))**(max(abs(i-x), abs(j-y)))):
    #                 Matrix[i,j] = phageID # dead S cell spots become Phages
    #                 self.PhageKills+=1
    #     return Matrix
    
    # def lysogeny(self, Matrix, X, Y, ID, LcellID):
    #     diffusivity = self.diffusivity       
    #     for x, y in zip(X,Y):
    #         # Find all S cells in the given radius
    #         leftend, rightend, bottom, top = self.CoordinateReturn(len(Matrix), x, y, diffusivity)
    #         Sindex_X, SindexY = np.where(Matrix[leftend:rightend+1,bottom:top+1]==ID)
            
    #         for i,j in zip(leftend+Sindex_X, bottom+SindexY):
    #             if(rand.random() <= abs(rand.gauss(self.StoDeathRate, self.StoDeathRate*self.MeanVariation))**(max(abs(i-x), abs(j-y)))):
    #                 Matrix[i,j] = LcellID # S cell spots become L cell
    #                 self.PhageKills+=1
    #     return Matrix
    

# Find where S cells are
# Count and return as many S cells as numbers Killed by lytic cycle
##### Include a condition as: ratio of no. of surrounding (diffusivity = 4) S cells/no. of phages.############
##### If ratio is more, then go lytic. If ratio is less, then go lysogeny ####################################
       
    def lyticlysogeny(self, Matrix, phageID, StoLRate = 0., StoDeathRate = 0.): # here ID is S cell ID
        X,Y = np.where(Matrix==phageID) # find phages
        #NoOfPhages = len(X) # coutning number of phages in grid
        
        if(len(X) <= 0 or StoLRate <= 0):
            return Matrix
        else:
            for x, y in zip(X,Y): # loop through selected phage IDs locations from index
                leftend, rightend, bottom, top = self.CoordinateReturn(len(Matrix), x, y, self.diffusivity)
                temp_NoOfSCells = np.count_nonzero(Matrix[leftend:rightend+1, bottom:top+1] == self.SCell.ID) # counting the number of S cells ID==1 around each dead C cells
                
                #print("calculating ratio")
                ratio = temp_NoOfSCells/(rand.randint(self.PhagesPerLDeath - 20, self.PhagesPerLDeath)) # Assuming each phage spot has around 80-100 phages
                #print("Ratio calculation done")
                
                if ratio > 0.50: # meaning S cells more, So lytic cycle
                    #Matrix = self.KillSCells(Matrix, X[index], Y[index], self.SCell.ID, self.Phage.ID)
                    #print("Ratio is " + str(ratio) + " So going Lytic")
                    #print("Lytic cycle decided")
                    NoOfConversions = np.sum(np.random.binomial(temp_NoOfSCells, abs(rand.gauss(self.StoDeathRate, self.StoDeathRate*self.MeanVariation)), self.PhagesPerLDeath)) # Need to improve upon this                    
                    if(NoOfConversions > 0 and temp_NoOfSCells > 0):                        
                        SindexX, SindexY = np.where(Matrix[leftend:rightend+1, bottom:top+1]==self.SCell.ID)
                        placepos = np.random.choice(np.arange(0, len(SindexX)), NoOfConversions)
                        #print(Matrix[leftend+SindexX[placepos],bottom+SindexY[placepos]])
                        Matrix[leftend+SindexX[placepos],bottom+SindexY[placepos]] = phageID
                        Matrix[x,y] = 0
                        self.StoDeathConversions+=NoOfConversions                         
                        #print("Lytic cycle done")
                else: # Do lysogeny
                    #Matrix = self.lysogeny(Matrix, X[index], Y[index], self.SCell.ID, self.LCell.ID)
                    #print("Ratio is " + str(ratio) + " So going Lysogeny")
                    #print("Lysogeny cycle decided")
                    NoOfConversions = np.sum(np.random.binomial(temp_NoOfSCells, abs(rand.gauss(self.StoLRate, self.StoLRate*self.MeanVariation)), self.PhagesPerLDeath)) # Need to improve upon this                    
                    if(NoOfConversions > 0 and temp_NoOfSCells > 0):                        
                        SindexX, SindexY = np.where(Matrix[leftend:rightend+1, bottom:top+1]==self.SCell.ID)
                        placepos = np.random.choice(np.arange(0, len(SindexX)), NoOfConversions)
                        #print(Matrix[leftend+SindexX[placepos],bottom+SindexY[placepos]])
                        Matrix[leftend+SindexX[placepos],bottom+SindexY[placepos]] = self.LCell.ID
                        #print("Lysogeny cycle done")
                        Matrix[x,y] = 0 # make phage spot to zero
                        
                        self.StoLConversions+=NoOfConversions                  
        return Matrix
 
                    
    def logger(self, name, delT, dNS, dNPhages, dNL, dNLg, N_LtoPhage, StoDeathConversions, N_LtoLg, StoLConversions):
        
        file = open(name + ".csv", "a")
        if(delT == 1):
            file.write("TimeStep,SCells,Phages,LCells,LgCells,dNS,dNC,dNLg,NoOfLtoPhage,NoOfStoDeath,NoOfLtoLg,NoOfStoL\n")
        file.write(str(delT) + "," + str(self.SCell.NtMinus) + "," + str(dNPhages) + "," + str(self.LCell.NtMinus) + "," + str(self.LgCell.NtMinus) + ","  + str(dNS) + "," + str(dNL) 
                   + "," + str(dNLg) + "," + str(N_LtoPhage) + "," + str(StoDeathConversions) + "," + str(N_LtoLg) + ","
                    + str(StoLConversions) + "," + "\n")
        
        file.close()
        
    
    
    def PrintInfo(self, Name):
        with open(Name+"Info.txt","w") as f:
            
            f.write("############## Starting the Simulation #################### \n \n")
            f.write("Parameters that are used for this simulation are: \n")
            data = [["Total Time ", self.T, 0], ["Time Steps ", self.dT, 0], 
                    ["Grid Size", np.shape(self.XYMatrix), 0], 
                    ["No Of Intial S Cells", self.SCell.N0, 0],
                    ["No Of Initial L Cells", self.LCell.N0, 0], 
                    ["Growth Rate for S Cells", self.SCell.r, 0],
                    ["Growth Rate for L Cells", self.LCell.r, 0],
                    ["Lambda S to L", self.StoLRate, self.MeanVariation*self.StoLRate],
                    ["Lambda L to PhageRate", self.LtoPhageRate, self.MeanVariation*self.LtoPhageRate],
                    ["Lambda S to DeathRate", self.StoDeathRate, self.MeanVariation*self.StoDeathRate],
                    ["L to Lg Rate", self.LtoLgRate, self.MeanVariation*self.LtoLgRate],
                    ["Growth rate for Lg", self.LgCell.r ,0],
                    ["Kill Radius (both sides)", self.diffusivity*2,0]]
            f.write(tabulate(data, headers=["Parameter Names", "Values", "Variations"]))
    
    
    def ReturnWholeNumber(self, dN, decimal):
        if(decimal>=1):
            return int(dN)+int(decimal), decimal-int(decimal)
        else:
            return int(dN), decimal + dN - int(dN)
        
    def saveframes(self, XYFrames, t):
        if(t%100!=0):
            return 0
        if(os.path.exists("Images")==False):
            os.mkdir("Images")
        
        foldername = "Images"
        
        fig, ax = plt.subplots(1,2,figsize=(9,3), gridspec_kw={'width_ratios': [2, 1]})
        plt.ion()
        names = ["S Cells", "L Cells", "Lg Cells"]
        colors1 = ["r-", "c-", "m-"]
        cmap = colors.ListedColormap(['k', 'red',"cyan", "magenta"])
        
        
        ax[0].plot(self.tlist, self.SCelllist, colors1[0], label = "{}".format(names[0]))
        #ax[0].plot(self.tlist, self.Phageslist, colors1[1], label = "{}".format(names[1]))
        ax[0].plot(self.tlist, self.LCelllist, colors1[1], label = "{}".format(names[1]))
        ax[0].plot(self.tlist, self.LgCelllist, colors1[2], label = "{}".format(names[2]))
        
        ax[0].set_xlim(0, self.T)
        ax[0].set_ylim(0, self.K + 750)
        ax[0].set_xlabel("Time (hours)", fontsize = 15)
        ax[0].set_ylabel("Population", fontsize = 15)
        ax[0].legend(fontsize = 20)
        mat = ax[1].imshow(XYFrames, origin="lower", cmap=cmap, vmin=0, vmax=3)
        ax[1].set_xlim(0, XYFrames.shape[0])
        ax[1].set_ylim(0, XYFrames.shape[1])
        ax[1].set_xticks(np.linspace(0, XYFrames.shape[0], 4))
        ax[1].set_yticks(np.linspace(0, XYFrames.shape[1], 4))
        cbar = plt.colorbar(mat, ax=ax[1], ticks = [0,1,2,3], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([" ", "S", "L", "Lg"])
        fig.tight_layout()
        plt.ioff()
        plt.savefig(os.path.join(os.getcwd(),foldername,"frame{}.png".format(t)),dpi=150)
        
        plt.close(fig)    
    
    def write_csv(self, name):
        Cells = {"SCells" : self.SCelllist, "LCells" : self.LCelllist, "LgCells" : self.LgCelllist}
        pd.DataFrame(data = Cells).to_csv(name + ".csv", index = False)
            
    
    
    def StartSimulation(self):

        print("############## Starting the Simulation ##################### \n \n ")
        print('''The Parameters used for running the simulation are: \n
        Total Time = {} \n
        Time steps = {} \n
        Matrix shape is = {} \n '''.format(self.T, self.dT, np.shape(self.XYMatrix)))

        XYMatrix = self.XYMatrix # Setting the Grid all zeros to start with
        loggername = "Logger"
        if(self.log and os.path.isfile("Logger.csv")):
            x = input("Removing the old log file is that ok ? yes/no : ")
            if(x=="yes"):
                os.remove("Logger.csv")
            else:
                sys.exit(0)
        SPopCounter = 0.0
        LPopCounter = 0.0
#       StoLCounter = 0.0 # There is No S to C at every step
        LtoPhageCounter = 0.0
        
        LgPopCounter = 0.0
        StoPhageCounter = 0.0
        
        LtoPhageCounter = 0.0
        LtoLgCounter = 0.0
        
        # Defining the 0th time parameters. Starting with SCell
        
        self.SCell.Nt = self.SCell.N0 # Setting Nt to N0. No logic here though
        XYMatrix = self.GenerateNo(XYMatrix, self.SCell.Nt, self.SCell.ID) # Generate Nt==N0 cells in the Grid
        self.SCell.NtMinus = self.SCell.Nt # Setting NtMinus to current Nt
        
        self.Phage.Nt = self.Phage.N0 # Setting Nt to N0. No logic here though
        XYMatrix = self.GenerateNo(XYMatrix, self.Phage.Nt, self.Phage.ID) # Generate Nt==N0 cells in the Grid
        self.Phage.NtMinus = self.Phage.Nt # Setting NtMinus to current Nt
        
        
        self.LCell.Nt = self.LCell.N0 # Setting Nt to N0. No logic here though
        XYMatrix = self.GenerateNo(XYMatrix, self.LCell.Nt, self.LCell.ID) # Generate Nt==N0 cells in the Grid
        self.LCell.NtMinus = self.LCell.Nt # Setting NtMinus to Current Nt
        #self.XYFrames.append(XYMatrix.copy()) #Append the XYGrid at t0 into the frame list

        self.LgCell.Nt = self.LgCell.N0 # Setting Nt to N0. No logic here though
        XYMatrix = self.GenerateNo(XYMatrix, self.LgCell.Nt, self.LgCell.ID) # Generate Nt==N0 cells in the Grid
        self.LgCell.NtMinus = self.LgCell.Nt # Setting NtMinus to Current Nt
        #self.XYFrames.append(XYMatrix.copy()) #Append the XYGrid at t0 into the frame list
        
        
        self.tlist.append(0)
        self.SCelllist.append(self.SCell.NtMinus)
        self.Phageslist.append(self.Phage.NtMinus)
        self.LCelllist.append(self.LCell.NtMinus)
        self.LgCelllist.append(self.LgCell.NtMinus)
        
        
        if(self.log): 
            self.saveframes(XYMatrix, 0)
            self.XYFrames.append(XYMatrix.copy())
        
        for t in range(1, int(self.T/self.dT)):
                       
            if(t%10==0): print("\n Running the time step {}".format(t))
            
############## Calculate the SCell Dynamics ###################################
            #print("LyticLysogeny decision making starts")
            #t0 = time.time()
            XYMatrix = self.lyticlysogeny(XYMatrix, self.Phage.ID, self.StoLRate, self.StoDeathRate)
            #print("LyticLysogeny decision taken")
            #t1 = time.time()
            #print("Time taken for decision and execution: " + str(t1-t0))
            # Logistic Growth of S                     
            self.SCell.Nt = self.LogisticGrowth(self.SCell.NtMinus, self.SCell.NtMinus, 
                                                self.LCell.NtMinus, self.LgCell.NtMinus, 
                                                self.SCell.r, self.SCell.K) # Now grow the Population
            
            dN_S = self.Calculate_dN(self.SCell.Nt, self.SCell.NtMinus) #Calculate the difference between Nt and NtMinus
            dN_SInt, SPopCounter = self.ReturnWholeNumber(dN_S, SPopCounter)
           
            
###################### Lg Cell dynamics ####################
         
            self.LgCell.Nt = self.LogisticGrowth(self.LgCell.NtMinus, self.SCell.NtMinus, 
                                                self.LCell.NtMinus, self.LgCell.NtMinus, 
                                                self.LgCell.r, self.LgCell.K) # Now grow the Population
            
            dN_Lg = self.Calculate_dN(self.LgCell.Nt, self.LgCell.NtMinus)
            dN_LgInt, LgPopCounter = self.ReturnWholeNumber(dN_Lg, LgPopCounter)
            
         
            # tempLtoLgRate = abs(rand.gauss(self.LtoLgRate, self.LtoLgRate*self.MeanVariation))
            # N_LtoLg = tempLtoLgRate*self.LCell.NtMinus
            # N_LtoLg, LtoLgCounter = self.ReturnWholeNumber(N_LtoLg, LtoLgCounter)
            # #N_LtoLg = np.random.binomial(self.LCell.NtMinus, self.LtoLgRate)
           
            
            N_LtoLg = np.random.binomial(self.LCell.NtMinus, abs(rand.gauss(self.LtoLgRate, self.LtoLgRate*self.MeanVariation))) # No of LgCells that are formed from L
            XYMatrix = self.Transmutation(N_LtoLg, XYMatrix, self.LgCell.ID, self.LCell.ID)
           
            
############## Calculate the LCell Dynamics ###################################            
            
            # Some of the LCells converts to Phage and kills
            # Transmutation from L to Phage
            
            N_LtoPhage = np.random.binomial(self.LCell.NtMinus, abs(rand.gauss(self.LtoPhageRate, self.LtoPhageRate*self.MeanVariation))) # No of LgCells that are formed from L
            XYMatrix = self.Transmutation(N_LtoPhage, XYMatrix, self.Phage.ID, self.LCell.ID)
          
            
            # Logistic Growth of L
            
            self.LCell.Nt = self.LogisticGrowth(self.LCell.NtMinus, self.SCell.NtMinus, 
                                                self.LCell.NtMinus, self.LgCell.NtMinus,
                                                self.LCell.r, self.LCell.K) # Now grow the Population
            
            
            dN_L = self.Calculate_dN(self.LCell.Nt, self.LCell.NtMinus) #Calculate the difference between Nt and NtMinus
            dN_LInt, LPopCounter = self.ReturnWholeNumber(dN_L, LPopCounter)        
           

######################## Filling in the Matrix for conversions and logistic Growths ##################
            
            if(dN_SInt>0):
                XYMatrix = self.PutPoints(XYMatrix, dN_SInt, self.SCell.ID) # add points if dN is positive
            else:
                XYMatrix = self.DelElements(XYMatrix, self.SCell.ID, -1*dN_SInt) # remove points if dN is negative
            
   
            if(dN_LInt>0):
                XYMatrix = self.PutPoints(XYMatrix, dN_LInt, self.LCell.ID) # add points if dN is positive
            else:
                XYMatrix = self.DelElements(XYMatrix, self.LCell.ID, -1*dN_LInt) # remove points if dN is negative
            
            
            if(dN_LgInt > 0):
                XYMatrix = self.PutPoints(XYMatrix, dN_LgInt, self.LgCell.ID) # add points if dN is positive
            else:
                XYMatrix = self.DelElements(XYMatrix, self.LgCell.ID, -1*dN_LgInt) # remove points if dN is negative
               
            
###################### Calculate all the NtMinus for S, C, Lg, Cg2, Cg3 cells #######################                                    
            
            self.SCell.NtMinus = np.count_nonzero(XYMatrix==self.SCell.ID) + dN_S - dN_SInt
            self.Phage.NtMinus = np.count_nonzero(XYMatrix==self.Phage.ID)
            self.LCell.NtMinus = np.count_nonzero(XYMatrix==self.LCell.ID) + dN_L - dN_LInt
            self.LgCell.NtMinus = np.count_nonzero(XYMatrix==self.LgCell.ID) + dN_Lg - dN_LgInt           
            
            self.tlist.append(t)
            self.SCelllist.append(int(self.SCell.NtMinus))
            self.Phageslist.append(int(self.Phage.NtMinus))
            self.LCelllist.append(int(self.LCell.NtMinus))
            self.LgCelllist.append(int(self.LgCell.NtMinus))
            
            
            if(self.log):
                self.logger(loggername, t, dN_S, self.Phage.NtMinus, dN_L, dN_Lg, N_LtoPhage, self.StoDeathConversions, N_LtoLg, self.StoLConversions)
                self.saveframes(XYMatrix, t)
                self.XYFrames.append(XYMatrix.copy())
                        
            self.PhageKills = 0 # resetting the logger variable
            self.StoDeathConversions = 0
            self.StoLConversions = 0 #resetting the logger variable
            self.Sdeaths = 0
