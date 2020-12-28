import numpy as np
import collections
import pandas as pd
import math
from io import StringIO
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import TextProtocol
#import MRKnnTrain
import heapq
import os
import ast


OUTPUT_PROTOCOL = TextProtocol

current = os.getcwd()
#counter = 0
class KNNTest(MRJob):
    '''
    KNN predicts classes. Receive test sets from the file and predict their classes based on the features of the test sets and compares 
    them with the real classes to see if the prediction is successful
    '''
    def show_tree(self, tree, total_width=60, fill=' '):
        """Pretty-print a tree.
        total_width depends on your input size"""
        output = StringIO()
        last_row = -1
        for i, n in enumerate(tree):
            if i:
                row = int(math.floor(math.log(i+1, 2)))
            else:
                row = 0
            if row != last_row:
                output.write('\n')
            columns = 2**row
            col_width = int(math.floor((total_width * 1.0) / columns))
            output.write(str(n).center(col_width, fill))
            last_row = row
        print (output.getvalue())
        print ('-' * total_width)
        return
        
    def configure_args(self):
        '''
        Input args. including the address of the model (output of KNNTrain), and the value of K.
        '''
        super(KNNTest,self).configure_args()
        #model's address
        self.add_passthru_arg("--model",
                                type = str,)
        #The value of k
        self.add_passthru_arg("-k",
                                type = str,
                                default = 3)
                                     
    def load_args(self,args):
        '''
        Reads the corresponding data based on the input args.
        '''
        super(KNNTest,self).load_args(args)
        #read model
        if self.options.model is None:
            #No input mod, error reported
            self.option_parser.error("please type a path")
        else:
            #read model
            self.model = {}
            #job = MRKnnTrain.KNNTrain()
            with open(current+'/'+self.options.model,encoding='utf-8') as src:
                for line in src:
                    # For each line of the model file, read the corresponding labels and features and store them in the dictionary.
                    label_model, features_model = line.split('\t')
                    features_model = features_model.replace('\n', '')#.strip('][').split(', ')
                    features_model = ast.literal_eval(features_model)
                    self.model[label_model] = features_model

        #read k values
        try:
            self.k = int(self.options.k)
        except:
            self.option_parser.error("K value must be integer.")

    def __init__(self, *args, **kwargs):
        super(KNNTest, self).__init__(*args, **kwargs)

    def steps(self): 
        return ([MRStep(mapper=self.mapper,reducer=self.reducer)])

    def mapper(self,_,line):
        '''
        Mapper function. Receives each row of the test set, extracts its feature set, and calculates the K points in the training set that are closest to it.
        The class with the most K points is determined and the class corresponding to that test ssample is predicted to be that class. 
        '''
        # Extract feature set and class of test data
        data = line.split(',')
        label = data[-1]
        features = [float(x) for x in data[:-1]] #austauschen durch lambda
        nearest = [] #k nearest points
        count = {} #The number corresponding to each category in nearest

        for cat in self.model:
            for point in self.model[cat]:
                # distance, multiplied by -1 because afterwards the heap sort will be used and needs to be ranked from largest to smallest, but the python implementation is the smallest heap, so *(-1)
                dis_euk = -1*np.linalg.norm(np.array(point)-np.array(features), ord=2) #L2 Norm/eukl Distanz
                dis_man = -1*np.linalg.norm(np.array(point)-np.array(features), ord=1) #L1 Norm/Manhatten Distanz
                dis_frobenius = -1*np.linalg.norm(np.array(point)-np.array(features)) #Default: Frobenius Distanz
                #Make a tuple of distances, points, and categories to which they belong for easy comparison
                item = tuple([dis_euk, point, cat])
                if(len(nearest)<self.k):
                    # If the nearest length is less than k, append directly
                    nearest.append(item)
                    continue
                elif(len(nearest)==self.k):
                    # If the nearest length is equal to k, transform the nearest into a heap
                    heapq.heapify(nearest)
                if(dis_euk > nearest[0][0]):
                    #If the distance of the new point is less than the longest point in the nearest, the longest point is popped out and the new point enters the nearest
                    heapq.heapreplace(nearest,item)        
        yield features, nearest
        #self.show_tree(nearest)

    def reducer(self, features, nearest):
        '''
        Reducer function that checks for suggestions for each line and writes them in a file
        '''
        nearest_list = []
        counter=0
        for near in nearest:
            nearest_list.append(near)

        #counter = counter+1

        yield features, nearest_list
       # yield counter, nearest_list

if __name__ == '__main__':
    KNNTest.run()
    self.show_tree(nearest)
