import numpy as np
import collections
import pandas as pd
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol, TextProtocol
import MRKnnTrain
import heapq
import os

OUTPUT_PROTOCOL = TextProtocol

current = os.getcwd()
true = 0
false = 0
class KNNTest(MRJob):
    '''
    KNN predicts classes. Receive test sets from the file and predict their classes based on the features of the test sets and compares 
    them with the real classes to see if the prediction is successful
    '''

    def configure_args(self):
        '''
        Input args. including the address of the model (output of KNNTrain), and the value of K.
        '''
        super(KNNTest,self).configure_args()
        #to-do: argument parser rausballern
        #model's address
        self.add_passthru_arg("--model",
                                type = str,
                                help = "the model path")
        #The value of k
        self.add_passthru_arg("-k",
                                type = str,
                                help = "the value of K",
                                default = 3)
                                     
    def load_args(self,args):
        '''
        Reads the corresponding data based on the input args.
        '''
        super(KNNTest,self).load_args(args)
        #read model
        if self.options.model is None:
            #No input mod, error reported
            self.option_parser.error("please type the path to the model.")
        else:
            #read model
            self.model = {}
            job = MRKnnTrain.KNNTrain()
            with open(current+'/'+self.options.model,encoding='utf-8') as src:
                for line in src:
                    # For each line of the model file, read the corresponding labels and features and store them in the dictionary.
                    label, features = job.parse_output_line(line.encode())
                    self.model[label] = features

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
        The class with the most K points is determined and the class corresponding to that test ssample is predicted to be that class. Then compare it with the real class, and if
        prediction is correct, output (true, 1), otherwise output (false, 1)
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
                dis_frobenius = -1*np.linalg.norm(np.array(point)-np.array(features)) #Default: Frobenius Distanz                #Make a tuple of distances, points, and categories to which they belong for easy comparison
                item = tuple([dis_euk, point, cat])
                if(len(nearest)<self.k):
                    # If the nearest length is less than k, append directly
                    nearest.append(item)
                    continue
                elif(len(nearest)==self.k):
                    # If the nearest length is equal to k, transform the nearest into a heap
                    heapq.heapify(nearest) #queueue statt heap?
                if(dis_euk > nearest[0][0]):
                    # If the distance of the new point is less than the longest point in the nearest, the longest point is popped out and the new point enters the nearest
                    heapq.heapreplace(nearest,item)
        # Calculate the category to which each point in nearest belongs
        for i in range(len(nearest)):
            temp = heapq.heappop(nearest)
            if(temp[2] not in count):
                count[temp[2]] = 1
            else:
                count[temp[2]] += 1
         # of most calculated categories        
        res = max(count, key=count.get) #optional: übergabe von threshhold
        if res: 
            yield data, res
        else: 
            res=["No result"]
            yield data, res

    def reducer(self, data, res):
        #label=str(label)

        res_list = []
        for element in res:
            res_list.append(element)
        yield data, res_list

if __name__ == '__main__':
    KNNTest.run()