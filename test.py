import os
import os.path
os.environ['KERAS_BACKEND']='tensorflow'
import csv
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.optimizers import Adam
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from mylib.models import metrics, losses
from keras.models import load_model
from keras.optimizers import SGD

x_test_path='./data/test'


def get_testdataset():
    x_return_test = []
    x_name = pd.read_csv("submit.csv") ['Id']
    for i in range(117):
        x_file_temp = os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_seg = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel*x_seg*0.9+x_voxel*0.1
        x_return_test.append(x_temp[34:66,34:66,34:66])
    return x_return_test


y_voxel = np.array(get_testdataset())
y_voxel = y_voxel.reshape(y_voxel.shape[0],32,32,32,1)
y_voxel = y_voxel.astype('float32')/255


#test
model=load_model("./tmp/best/weights.h5", compile=False)
result = model.predict(y_voxel,batch_size = 1)
#np.savetxt("E:/DenseSharp2/tmp/resultfortrick/submit3.csv", result, delimiter = ',')
#np.save("E:/DenseSharp2/tmp/resultfor66/result.npy" , result[0])
#result = np.load("E:/DenseSharp2/tmp/resultforfifth/result.npy" )
print(result)
csv = pd.read_csv("./data/test.csv")
csv.iloc[:, 1] = result[:, 1]
csv.columns = ['Id', 'Predicted']

csv.to_csv("./tmp/best/submission.csv", index=None)



