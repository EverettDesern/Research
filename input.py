'''
Created on Jun 10, 2019

@author: edesern
'''

import png
import pickle
import json
import csv
import numpy as np
 
if __name__ == '__main__':
    List = []
    vid = input("Enter the video link")
    # /Users/edesern/Downloads/out.dat
    f = open(vid, "rb")
    height = int(input("Enter the height of the video"))
    width = int(input("Enter the width of the video"))
    blockSi = int(input("Enter the block size"))
    searchReg = int(input("Enter the search region"))

    def main():
        # reads in pixel values to a list of frames (2D arrays)
        for x in range(300):
            a = [[0 for w in range(width)] for h in range(height)]
            for i in range(width):
                for j in range(height):
                    b = f.read(1)
                    num = ord(b)
                    a[i][j] = num
                    
            List.append(a)
        
    def search(blockSize, searchRegion):
        # gets block from the current frame and iterates through the prior frame
        # to find the best MSE
        blockNumber = 0
        num1 = 0
        whole = []
        regionSize = blockSize * searchRegion

        # fix
        if(regionSize > 64):
            regionSize = 64
        sub = regionSize - blockSize
        if sub % 2 == 1:
            div = sub / 2 + 1
        else:
            div = sub / 2
                        
        for i in range(2,300):
            print(i)
            for j in range(0, width, blockSize):
                for p in range(0, height, blockSize):
                    # calls the block function to extract a macroblock
                    # of size blockSize*blockSize
                    # this is a macroblock of the current frame
                    if(p > height - blockSize):
                        continue
                    if(j > width - blockSize):
                        continue
                    a = block(blockSize, j, p, i)
                    blockNumber += 1
                    errorList = []
                    xList = []
                    yList = []
                    min = 100000
                        
                    if p - div < 0:
                        continue
                        #lowY = 0
                    else:
                        lowY = p - div
                        
                    if p + div + blockSize > 64:
                        continue
                        #highY = 64
                    else:
                        highY = p + div + blockSize
                        
                    if j - div < 0:
                        continue
                        #lowX = 0
                    else:
                        lowX = j - div
                        
                    if j + div + blockSize > 64:
                        continue
                        #highX = 64
                    else:
                        highX = j + div + blockSize
                        
                    for s in range(int(lowX), int(highX-blockSize+1)):
                        for d in range(int(lowY), int(highY-blockSize+1)):
                            # calls the block function to extract a 
                            # macroblock from the prior frame
                            n1Block = block(blockSize, s, d, i-1)
                            #n2Block = block(blockSize, s, d, i-2)
                            # calculates the mean square error of the
                            # current frame macroblock and the prior
                            # frame macroblock
                            meanSquareError = compare(n1Block, a, blockSize)
                            if meanSquareError < min:
                                min = meanSquareError
                            xList.append(s)
                            yList.append(d)
                            errorList.append(meanSquareError)
                    prev1Block = region(i, lowX, highX, lowY, highY, regionSize)
                    prev2Block = region(i-1, lowX, highX, lowY, highY, regionSize)
                    prevList = []
                    prevList.append(prev1Block)
                    #prevList.append(prev2Block)
                    # threshold for MSE "match"
                    # gets the index of the "match"
                    index = errorList.index(min)
                    indexX = xList[index]
                    indexY = yList[index]
                    block1 = [x for row in prev1Block for x in row]
                    block2 = [x for row in prev2Block for x in row]
                    arraySize = int(div * 2 + 1)
                    if(min > 1):
                        xArray = [0.0] * (arraySize + 1)
                        yArray = [0.0] * (arraySize)
                        xArray[0] = 1.0
                    else:
                        num1 += 1
                        #print(num1)
                        xArray = [0.0] * (arraySize + 1)
                        # yArray = np.zeros(arraySize, dtype = int)
                        yArray = [0.0] * (arraySize)
                        xValue = int((j-indexX) + div + 1)
                        yValue = int((p-indexY) + div)
                        xArray[xValue] = 1.0
                        yArray[yValue] = 1.0
                    # tup = block1 + block2 + [j-indexX, p-indexY]
                    tup = block1 + block2 + xArray + yArray
                    whole.append(tup)
        #print(len(whole))

        
    
        with open('train.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            for i in whole:
                writer.writerow(i)
        
    #    with open('train.csv', newline='') as f:
    #        reader = csv.reader(f)
    #        for row in reader:
    #             print(row)

    
    # function for extracting a block of size blockSize*blockSize
    def block(blockSize, x, y, frame):
        currentFrame = List[frame]
        macroBlock = [[0 for d in range(blockSize)] for q in range(blockSize)]
        l = 0
        h = 0
        lowRangeX = x
        highRangeX = x + blockSize
        lowRangeY = y
        highRangeY = y + blockSize
        
        for v in range(lowRangeX, highRangeX):
            h = 0
            for b in range(lowRangeY, highRangeY):
                value = currentFrame[v][b]
                macroBlock[l][h] = value
                h += 1
            l += 1
        return macroBlock              
                        
    # function for calculating the mean square error between two blocks
    def compare(first, second, blockSize):
        MSE = 0
        sum = 0
        for i in range(blockSize):
            for j in range(blockSize):
                value1 = first[i][j]
                value2 = second[i][j]
                diff = value2 - value1
                sum += (diff * diff)
        MSE = sum / (blockSize * blockSize)
        actual = MSE
        return actual
    
    def region(frame, lowX, highX, lowY, highY, regionSize):
        prev1 = List[frame-1]
        block1 = [[0 for d in range(regionSize)] for q in range(regionSize)]
        l = 0
        h = 0
        for v in range(int(lowX), int(highX)):
            h = 0
            for b in range(int(lowY), int(highY)):
                value1 = prev1[v][b]
                block1[l][h] = value1
                h += 1
            l += 1
        return block1
                       
                    
    

main()
search(blockSi, searchReg)   
                        
                        
                        
                        
                        
                        
                        