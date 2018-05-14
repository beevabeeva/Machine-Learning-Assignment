
#                        .-'~~~-.
#                      .'o  oOOOo`.
#                     :~~~-.oOo   o`.
#                      `. \ ~-.  oOOo.
#                        `.; / ~.  OO:
#                        .'  ;-- `.o.'
#                       ,'  ; ~~--'~
#                       ;  ;
# _______\|/__________\\;_\\//___\|/________





from collections import Counter
from sklearn.metrics import confusion_matrix
import csv
import re
import random
import numpy as np
import matplotlib.pylab as plt
import itertools
from math import *
       
def nb_analysis(training, testing, laplace):
# Where training := training data subset
#       testing  := testing data subset
#       laplace   := turn on laplace smoothing (1= on, 0 = off)

#split mushrooms into poisonous and edible
    num_poisonous = 0
    poisonous_features = []
    num_edible = 0
    edible_features = []
    
    for r in training:
        if r[0][0] == "p":
            num_poisonous = num_poisonous + 1
            poisonous_features.extend(list(set(re.split("\s+",r[0])[1:])))
        else:
            num_edible = num_edible + 1
            edible_features.extend(list(set(re.split("\s+",r[0])[1:])))
    
    # if removeStopWords == 1:
    #     negative_words = [word for word in negative_words if len(word) >= 3]
    #     positive_words = [word for word in positive_words if len(word) >= 3
]
#get counts for each feature
    poison_count = Counter(poisonous_features)
    edible_count = Counter(edible_features)
    
#list of all features used in the training mushrooms
    all_features = list(set(poisonous_features + edible_features))

#prob of poisonous/edible mushroom (Priors)
    prob_edible = num_edible/len(training)
    prob_poison = num_poisonous/len(training)

    
#class prediction function
    def class_prediction(mushroom):
        poison = 1
        edible = 1
        mushroom_features = list(re.split("\s+",mushroom)[1:])
        
        # if removeStopWords == 1:
        #     review_words = [word for word in review_words if len(word) >= 3]
    
        k = 0
        n = 0
        if laplace == 1:
            k = 1
            n = 2
            
        for value in all_features:
            # if value not in mushroom_features:
            #     edible = edible*(1-((edible_count.get(value,0) + k)/(num_edible + n)))
            #     poison = poison*(1-((poison_count.get(value,0) + k)/(num_poisonous + n)))
            # else:
                edible = edible*((edible_count.get(value,0) + k)/(num_edible + n))
                poison = poison*((poison_count.get(value,0) + k)/(num_poisonous + n))

        #Class Conditional * Prior:
        edible = edible*prob_edible
        poison = poison*prob_poison

        if poison > edible:
            return -1
        return 1

    y_true = []
    y_pred = []

#find class predictions for each mushroom in testing data, then display each mushroom with its prediction
    for t in testing:
        print("Mushroom:", t[0])
        if t[0][0] == "p":
            y_true.append(-1)
        else:
            y_true.append(1)

        if class_prediction(t[0]) == 1:
            print("Decision: edible\n")
            y_pred.append(1)
        else:
            print("Decision: poisonous\n")
            y_pred.append(-1)


    cm = confusion_matrix(y_true, y_pred)

#plot confusion matrix
    title = "Confusion Matrix"
    classes = ['poisonous','edible']
    cmap=plt.cm.BuGn
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if cm.max() > 1:
        thresh = cm.max()/2
    else:
        thresh = 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#find accuracy of predictions
    numCorrect = (cm[0][0]+cm[1][1])
    if numCorrect == 0:
        print("Accuracy: 0%")
    else:
        print("Accuracy: ", int(numCorrect/len(testing)*100), "%")
    



#split mushrooms randomly into training and testing data
with open("agaricus-lepiota.data", 'r') as file:
    mushrooms = list(csv.reader(file))

trainingData = random.sample(mushrooms, int(len(mushrooms)*2/3))
testingData = []
for mushroom in mushrooms:
    if mushroom not in trainingData:
        testingData.append(mushroom)

nb_analysis(trainingData,testingData,1)