import time
import json
import requests
import numpy as np
from tensorflow import keras

def print_performance(max_iter=10):
    # test image set
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    test_images = np.expand_dims(test_images/255.0, -1).tolist() 
    
    headers = {"content-type": "application/json"}
    url = "http://localhost:8080/predict"
    images = json.dumps({"data": test_images})
    
    response_times = list()
    for idx in range(max_iter):
        start_time = time.time()
        response = requests.post(url, data=images, headers=headers)
        response_times.append(time.time() - start_time)
        predictions = response.json()['predict']
    print(f'AVG = {np.average(response_times):>.3f} s, STD = ±{np.std(response_times):>.3f} s')
    

if __name__ == '__main__':
    print_performance()