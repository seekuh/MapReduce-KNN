# -*- coding: UTF-8 -*-
import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol

class KNNTrain(MRJob):

    OUTPUT_PROTOCOL = JSONProtocol

    def __init__(self, *args, **kwargs):
        super(KNNTrain, self).__init__(*args, **kwargs)
        
    def steps(self):
        return ([MRStep(mapper=self.mapper,reducer=self.reducer)])

    def mapper(self,_,line):
        data = line.split(',')
        yield data[-1], data[:-1]

    def reducer(self, label, features):
        features_list = []
        for feature in features:
            feature = [float(x) for x in feature]
            features_list.append(feature)
        yield label, features_list

if __name__ == '__main__':
    KNNTrain.run()