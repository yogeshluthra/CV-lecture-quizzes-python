import numpy as np
import scipy.signal as sp


def find_template_1D(t, s):
    corr1d=np.zeros(s.size-t.size+1)
    for i in range(corr1d.size):
        corr1d[i]=np.sum(t[:]*s[i:i+t.size])
    return np.argmax(corr1d)

s = np.array([-1, 0, 0, 5, 1, 1, 0, 0, -1, -7, 2, 1, 0, 0, -1])
t = np.array([-1, -7, 2])

print "Signal: \n {} \n {}".format(np.array(range(s.size)), s)
print "Template: \n {} \n {}".format(np.array(range(t.size)), t)

index = find_template_1D(t, s)
print "Index: {}".format(index)
