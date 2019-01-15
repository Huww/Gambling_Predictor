import numpy as np

def calculate_accuracy(predictions, test_labels, test_features):
    correct = 0
    results=[]

    for i in range(0, len(predictions)):
        probabilities = predictions[i]['probabilities']
        index = np.argmax(probabilities)
        result = ''
        if index == 0:
            result = 'H'
        elif index == 1:
            result = 'D'
        else:
            result = 'A'

        if result == test_labels[i]:
            correct+=1
        results.append(result)
    accuracy = correct / len(predictions)
    print("Total accuracy from {} tests is: {}".format(len(predictions), accuracy))
    print(results)
    print(test_labels)
