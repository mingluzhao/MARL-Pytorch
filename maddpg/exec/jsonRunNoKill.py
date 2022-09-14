import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
import numpy as np
import itertools as it
import argparse

class ExcuteCodeOnConditionsParallel:
    def __init__(self, codeFileName, numSample, numCmdList):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, conditions):
        assert self.numCmdList >= len(conditions), "condition number > cmd number, use more cores or less conditions"
        process = []
        for condition in conditions:
            cmd = ['python3', self.codeFileName] + [str(v) for v in condition.values()] # [str(x) for y in zip(var_names, values) for x in y]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            process.append(proc)

        for proc in process:
            proc.communicate()

def main():
    condition_dict = {"num_predators": [3, 4, 5, 6],
                      "speed": [1],
                      "cost": [0, 0.01, 0.02, 0.03],
                      "selfish": [0, 1]}

    var_names, values = zip(*condition_dict.items())
    conditions = [dict(zip(var_names, v)) for v in it.product(*values)]

    startTime = time.time()
    fileName = 'run_nokill_handcrafted_hunting.py'
    excuteCodeParallel = ExcuteCodeOnConditionsParallel(fileName, numSample = None, numCmdList = int(0.8 * os.cpu_count()))
    excuteCodeParallel(conditions)
    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))

if __name__ == '__main__':
    main()
