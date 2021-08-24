import torch
import timeit
import numpy as np
import time

cuda0 = torch.device('cuda')

h = torch.ones((3,3,3,3), dtype=torch.float32, device=cuda0)
h[:, :, 1,1] = 0.0

x_32 = torch.tensor(np.random.random_sample((1, 3, 32, 32)), dtype=torch.float32, device=cuda0)
x_32batch = torch.tensor(np.random.random_sample((32, 3, 32, 32)), dtype=torch.float32, device=cuda0)
x_320 = torch.tensor(np.random.random_sample((1, 3, 256, 256)), dtype=torch.float32, device=cuda0)

    
def conv_smol():
    torch.nn.functional.conv2d(x_32, h)
    
def conv_batch():
    torch.nn.functional.conv2d(x_32batch, h)

def conv_fat():
    torch.nn.functional.conv2d(x_320, h)

print("conv_smol time")
smolTimes = []
for i in range(0, 5):
    print(i)
    smolTimes.append(timeit.timeit(conv_smol, number=100000))
    print("32x32", smolTimes[i])
print("avg", sum(smolTimes) / len(smolTimes))
time.sleep(5)
print("conv_batch time")
batchTimes = []
for i in range(0, 5):
    print(i)
    batchTimes.append(timeit.timeit(conv_batch, number=100000))
    print("32x32batch", batchTimes[i])
print("avg", sum(batchTimes) / len(batchTimes))
time.sleep(5)   
print("conv big time")
bigTimes = []
for i in range(0, 5):
    print(i)
    bigTimes.append(timeit.timeit(conv_fat, number=100000))
    print("256x256", bigTimes[i])
print("avg", sum(bigTimes) / len(bigTimes))
