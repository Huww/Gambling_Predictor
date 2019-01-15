import dataset
import pickle


data = dataset.Dataset("/Users/Huw/Documents/GitHub/Gambling_Predictor/data/book2.csv",5)
for result in data.processed_results:
    print(result)
