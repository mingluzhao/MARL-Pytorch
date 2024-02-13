import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
import numpy as np
import itertools as it
import json
import base64

# Function to encode the condition dictionary to a Base64 string
def encode_condition(condition):
    json_string = json.dumps(condition)
    base64_bytes = base64.b64encode(json_string.encode('utf-8'))
    return base64_bytes.decode('utf-8')


class ExecuteCodeOnConditionsParallel:
    def __init__(self, codeFileName, numCmdList):
        self.codeFileName = codeFileName
        self.numCmdList = numCmdList

    def __call__(self, conditions):
        assert self.numCmdList >= len(conditions), "More conditions than available commands. Reduce conditions or increase available commands."
        
        conda_path = "/home/vi3850-64core1/miniconda3/bin/conda"  # Path to the conda executable
        cmdList = [
            f'bash -c "{conda_path} run -n marl python {self.codeFileName} {encode_condition(condition)}"'
            for condition in conditions
        ]        
        
        processList = []
        for cmd in cmdList:
            process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, executable='/bin/bash')
            processList.append(process)
        
        for i, proc in enumerate(processList):
            stdout, stderr = proc.communicate()
            print(f"Output for condition {conditions[i]}:\n{stdout.decode()}\n")
            if proc.returncode != 0:
                print(f"Error for condition {conditions[i]}:\n{stderr.decode()}\n")
            else:
                print(f"Condition {conditions[i]} completed successfully.\n")

        return cmdList


def main():
    condition = {
        "numSheepsLevels": [1, 2, 4], 
        "sheepSpeedMultiplierLevels": [1.1], #[0.7, 1.1, 1.5], 
        "individualRewardWolfLevels": [0, 1], 
        "discrete_actionLevels": [0],
        "seedLevels": [101, 1001, 10001, 100001, 1000001],
        "killZoneRatioLevels": [1.2, 1.5, 1.8]
        }

    numSheepsLevels = condition['numSheepsLevels']
    sheepSpeedMultiplierLevels = condition['sheepSpeedMultiplierLevels']
    individualRewardWolfLevels = condition['individualRewardWolfLevels']
    discrete_actionLevels = condition['discrete_actionLevels']
    seedLevels = condition['seedLevels']
    killZoneRatioLevels = condition['killZoneRatioLevels']

    startTime = time.time()
    fileName = 'train_handcrafted_env.py'
    numCpuToUse = int(0.9 * os.cpu_count())
    excuteCodeParallel = ExecuteCodeOnConditionsParallel(fileName, numCpuToUse)
    print("start")

    conditionLevels = [(numSheeps, sheepSpeedMultiplier, individualRewardWolf, discrete_action, seed, killZoneRatio)
                       for numSheeps in numSheepsLevels
                       for sheepSpeedMultiplier in sheepSpeedMultiplierLevels
                       for individualRewardWolf in individualRewardWolfLevels
                       for discrete_action in discrete_actionLevels
                       for seed in seedLevels
                       for killZoneRatio in killZoneRatioLevels]

    conditions = []
    for condition in conditionLevels:
        numSheeps, sheepSpeedMultiplier, individualRewardWolf, discrete_action, seed, killZoneRatio = condition
        parameters = {'numSheeps': numSheeps, 'sheepSpeedMultiplier': sheepSpeedMultiplier, 'individualRewardWolf': individualRewardWolf,
                      'discrete_action': discrete_action, 'seed': seed, 'killZoneRatio': killZoneRatio}
        conditions.append(parameters)

    cmdList = excuteCodeParallel(conditions)
    print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))

if __name__ == '__main__':
    main()

