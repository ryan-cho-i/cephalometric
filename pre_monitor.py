from concurrent.futures import ThreadPoolExecutor
from time import *
import os
import psutil

class MemoryMonitor () :

    def __init__(self, name):

        self.working = True
        self.name = name
        self.cpu=[]

    def measuring (self) : 

        # Record the used memory of CPU & GPU per seconds

        while self.working :            
            os.system("gpustat >> " + self.name)

            mem = psutil.virtual_memory()
            MB = mem.used / (2 ** 20)
            self.cpu.append(MB)

            sleep(1)

<<<<<<< HEAD
class MemoryMonitor2 () :
    
    def __init__(self):

        self.working = True
        self.cpu=[]

    def measuring (self) : 

        # Record the used memory of CPU & GPU per seconds

        while self.working :            

            mem = psutil.virtual_memory()
            MB = mem.used / (2 ** 20)
            self.cpu.append(MB)

            sleep(1)            

=======
>>>>>>> e45329e9a9f7e9da5b669037c4a395b5315d8463
# Extract the max of GPU Memory 

def gpu_max (name):
    f = open (name)
    lines = f.readlines()
    value = []
    i=0
    for i in range(len(lines)) : 
        if i%2 == 1:
            value.append(lines[i].split('/')[0].split('|')[2])

    return max(value)

# Record the used memory of CPU & GPU while operating function

<<<<<<< HEAD
def Monitor (func, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, DEVICE, name) :
=======
def Monitor (func, model_dir, x_test_dir, y_test_dir, preprocessing_fn, CLASSES, DEVICE, name) :
>>>>>>> e45329e9a9f7e9da5b669037c4a395b5315d8463

    with ThreadPoolExecutor(max_workers=2) as executor: 
        monitor = MemoryMonitor(name)
        executor.submit(monitor.measuring)

        try : 
<<<<<<< HEAD
            main_thread = executor.submit(func, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, DEVICE)
            first, done, m, prob = main_thread.result()

        finally:
            monitor.working = False 

    return monitor.cpu, first, done, m, prob

# Only Record the used memory of CPU while operating function

def Monitor2 (func, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, DEVICE) :

    with ThreadPoolExecutor(max_workers=2) as executor: 
        monitor = MemoryMonitor2()
        executor.submit(monitor.measuring)

        try : 
            main_thread = executor.submit(func, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, DEVICE)
            first, done, m, prob = main_thread.result()
=======
            main_thread = executor.submit(func, model_dir, x_test_dir, y_test_dir, preprocessing_fn, CLASSES, DEVICE)
            prob, first, done, m = main_thread.result()
>>>>>>> e45329e9a9f7e9da5b669037c4a395b5315d8463

        finally:
            monitor.working = False 

<<<<<<< HEAD
    return monitor.cpu, first, done, m, prob
=======
    return monitor.cpu, first, done, m,  
>>>>>>> e45329e9a9f7e9da5b669037c4a395b5315d8463
