import csv

class DataSet():
    def __init__(self,filename):
        filestream = open(filename,'r')
        csv_reader = csv.reader(filestream,delimiter=',')
        self.raw_inputs = [[int(r) for r in row] for row in csv_reader]
    def getRaw(self):
        return self.raw_inputs
    def getInputs(self):
        return [row[:-1]for row in self.getRaw()]
    def getTargets(self):
        return [[x[-1]] for x in self.raw_inputs]
    def getNumInputElem(self):
        return len(self.getInputs()[0])