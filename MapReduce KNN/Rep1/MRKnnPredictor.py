import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol
import KNN
import heapq
import os

current = os.getcwd()
true = 0
false = 0
class KNNTest(MRJob):


    def configure_args(self):

        super(KNNTest,self).configure_args()

        self.add_passthru_arg("--model",
                                type = str,
                                help = "the model path")
        #K的值
        self.add_passthru_arg("-k",
                                type = str,
                                help = "the value of K",
                                default = 3)
                                     
    def load_args(self,args):
        super(KNNTest,self).load_args(args)

        if self.options.model is None:
            self.option_parser.error("please type the path to the model.")
        else:
            self.model = {}
            job = KNN.KNNTrain()
            with open(current+'/'+self.options.model,encoding='utf-8') as src:
                for line in src:
                    #对model文件的每一行，读取相应的label和features，并存储到字典中。
                    label, features = job.parse_output_line(line.encode())
                    self.model[label] = features

        #读取K值。
        try:
            self.k = int(self.options.k)
        except:
            self.option_parser.error("K value must be integer.")

    def __init__(self, *args, **kwargs):
        super(KNNTest, self).__init__(*args, **kwargs)

    def steps(self):
        return ([MRStep(mapper=self.mapper,reducer=self.reducer)])

    def mapper(self,_,line):
        data = line.split(',')
        label = data[-1]
        features = [float(x) for x in data[:-1]]
        nearest = [] 
        count = {} 

        for cat in self.model:
            for point in self.model[cat]:
                dis = -1*np.linalg.norm(np.array(point)-np.array(features)) 
                item = tuple([dis, point, cat])
                if(len(nearest)<self.k):
                    nearest.append(item)
                    continue
                elif(len(nearest)==self.k):
                    heapq.heapify(nearest)
                if(dis > nearest[0][0]):
                    heapq.heapreplace(nearest,item)
        for i in range(len(nearest)):
            temp = heapq.heappop(nearest)
            if(temp[2] not in count):
                count[temp[2]] = 1
            else:
                count[temp[2]] += 1
        res = max(count, key=count.get)
        if(res==label):
            yield 'true', 1
        else:
            yield 'false', 1

    def reducer(self, label, num):
        if False: yield
        if(label=='true'):
            global true
            true = sum(num)
        else:
            global false
            false = sum(num)

if __name__ == '__main__':
    KNNTest.run()
    print("Accuary:"+str(true/(true+false)*100)+"%")