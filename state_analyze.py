import pickle
from lanelines.state import State
import matplotlib.pyplot as plt
import numpy as np

with open('state_file.pkl', 'rb') as state_file:
    state = pickle.load(state_file)

#print(len(state._error_images))
#print(state._error_images[0].shape[0]/256)
#plt.imshow(np.reshape(state._error_images[0], (256, 576)))
#plt.show()
ns = State()

ll = []
rr = []
for item in state._measurements:
    #print(item[0], item[1])
    ns.add_measurements(item[0][0], item[1][0])
    #print(ns.left_lane)
    ll.append(ns.left_lane)
    rr.append(ns.right_lane)

ll = np.array(ll)
rr = np.array(rr)

plt.figure()
print(ll[:, 0])
plt.plot(ll[:, 0], 'b.')
plt.plot(rr[:, 0], 'r.')
plt.figure()
plt.plot(ll[:, 1], 'b.')
plt.plot(rr[:, 1], 'r.')
plt.figure()
plt.plot(ll[:, 2], 'b.')
plt.plot(rr[:, 2], 'r.')

plt.show()


