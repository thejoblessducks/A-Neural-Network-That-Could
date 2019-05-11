import csv

class DataSet():
    #This class reads and organizes a CSV file (can be a .txt file)
    def __init__(self,filename):
        filestream = open(filename,'r')
        csv_reader = csv.reader(filestream,delimiter=',')
        self.raw_inputs = [[int(r) for r in row] for row in csv_reader]
    def getRaw(self):
        return self.raw_inputs
    def getInputs(self,test=False):
        if not test: 
            #File is an input to network, ignores last column (target)
            return [row[:-1]for row in self.getRaw()]
        #File given is a testing file last column is part of test entry
        return self.getRaw()
    def getTargets(self):
        #Returns matrix of all the targets for all the input entries/examples
        return [[x[-1]] for x in self.raw_inputs]
    def getNumInputElem(self):
        #Returns the number of columns in an input entry/file
        return len(self.getInputs()[0])