import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol
import MRKnnTrain
import heapq
import os
import json
from itertools import islice
import ast
import csv
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix


startTime = datetime.now()

k = 10 
model = {}
df_new = pd.DataFrame()
with open('./model_neu.json',encoding='utf-16') as src:
    for line in src:
        label_model, features_model = line.split('\t')
        features_model = features_model.replace('\n', '')
        features_model = ast.literal_eval(features_model)
        label_model = ast.literal_eval(label_model)
        model[label_model] = features_model

#print(model.items()[0])
#print(list(model.items())[0])
#model_neu = json.loads(model)
#print(list(model_neu.items())[0])
#print(list(model.values())[0])
#file = pd.read_csv('.\data.csv')
with open("./test_neu.csv") as src:
    predictor_true = 0
    predictor_false = 0
    y_pred=[]
    for line in src:
        # Extract feature set and class
        data = line.split(',')
        label = data[-1]
        #print(data[:-1])
        print(data[-1])
        features = [float(x) for x in data[:-1]]
        #print(features)
        nearest = [] #k nearest points
        count = {} #The number corresponding to each category in nearest

        for cat in model:
            for point in model[cat]:
                # distance, multiplied by -1 because afterwards the heap sort will be used and needs to be ranked from largest to smallest, but the python implementation is the smallest heap, so *(-1)
                #print(point)
                point[1:] = [float(x) for x in point[1:]]
                #print(np.array(point[1:]))
                #print(np.array(features))
                dis = -1*np.linalg.norm(np.array(point[1:])-np.array(features)) 
                #Make a tuple of distances, points, and categories to which they belong for easy comparison
                item = tuple([dis, point[1:], cat, point[0]])
                if(len(nearest)<k):
                    # If the nearest length is less than k, append directly
                    nearest.append(item)
                    continue
                elif(len(nearest)==k):
                    # If the nearest length is equal to k, transform the nearest into a heap
                    heapq.heapify(nearest)
                if(dis > nearest[0][0]):
                    # If the distance of the new point is less than the longest point in the nearest, the longest point is popped out and the new point enters the nearest
                    heapq.heapreplace(nearest,item)
        
        heaptemp = heapq.heappop(nearest)
        #print(range(len(nearest)))

        for i in range(len(nearest)):
            temp = heapq.heappop(nearest)
            #print(heapq.heappop(nearest))
            #print(temp[2])
            #print(count)
            if(temp[2] not in count):
                count[temp[2]] = 1
            else:
                count[temp[2]] += 1
            #print(count[temp[2]])
        # of most calculated categories        
        res = max(count, key=count.get)
        y_pred.append(res)
        # Output true if the prediction is successful, otherwise false
        if(res==label):
            predictor_true += 1
        else:
            predictor_false += 1

        print(predictor_true)
        print(predictor_false)
        #temp = heapq.heappop(nearest)
        # print(temp)
        # for neighbour in nearest:
        #     #print(neighbour)
        #     #print(neighbour[3])
        #     df_new = df_new.append(file[(file['id'] == neighbour[3])])
    
print("Accuary:"+str(predictor_true/(predictor_true+predictor_false)*100)+"%")
print(datetime.now() - startTime)
# print(df_new)

# print('Ihre Songvorschläge basierend auf Ihrer Eingabe: \n')

# for index, song in df_new.iterrows():
#     print('Name des Songs: {} \n Künstler: {} \n'.format(song['name'], song['artists']))

# print(datetime.now() - startTime)