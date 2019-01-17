import numpy as np

def calculate_accuracy(predictions, test_labels, test_features):
    correct = 0
    results=[]
    bank = 100
    bets=0
    bets_won = 0

    for i in range(0, len(predictions)):
        probabilities = predictions[i]['probabilities']
        index = np.argmax(probabilities)
        result = ''
        if index == 0:
            result = 'H'
            key = 'odds-home'
        elif index == 1:
            result = 'D'
            key = 'odds-draw'
        else:
            result = 'A'
            key = 'odds-away'

        if result == test_labels[i]:
            correct+=1
            if probabilities[index] > 0.65:
                bets+=1
                bets_won += 1
                bank+=test_features[key][i] - 1
        else:
            if probabilities[index] > 0.65:
                bets+=1
                bank-=1
        results.append(result)


    accuracy = correct / len(predictions)
    print("Total accuracy from {} tests is: {}".format(len(predictions), accuracy))
    print(results)
    print(test_labels)
    print(bank)
    print(bets)
    print(bets_won)
