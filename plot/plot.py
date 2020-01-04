from matplotlib import pyplot as plt
from matplotlib import numpy as np

plt.plot(np.loadtxt('acc.txt'), color='blue', label='acc')
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(np.loadtxt('loss.txt'), color='blue', label='loss')
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train'], loc='upper left')
plt.show()