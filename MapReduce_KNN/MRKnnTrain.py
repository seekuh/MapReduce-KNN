# -*- coding: UTF-8 -*-
import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol
import ast
from datetime import datetime


startTime = datetime.now()


class KNNTrain(MRJob):

    '''
    KNN training class. Receive training set input, output (type, list of feature sets).
    '''

    OUTPUT_PROTOCOL = JSONProtocol

    def __init__(self, *args, **kwargs):
        super(KNNTrain, self).__init__(*args, **kwargs)
        
    def steps(self):
        return ([MRStep(mapper=self.mapper,reducer=self.reducer)])

    def mapper(self,_,line):
        '''
        Mapper function that takes the rows of the training set, distinguishes the types in the rows from the feature set, and outputs (type, feature set)
        '''
        data = line.split(',')
        yield data[-1], (data[:-1])

    def reducer(self, label, features):
        '''
        Reducer function that takes Mapper output and concatenates feature sets of the same type, output (type, list of feature sets)
        '''
        features_list = []
        #datapoints = features[1:]
        for feature in features:
            #feature = [ast.literal_eval(x) for x in feature]
            features_list.append(feature)
        yield label, features_list

if __name__ == '__main__':
    startTime = datetime.now()
    KNNTrain.run()
    print(datetime.now() - startTime)