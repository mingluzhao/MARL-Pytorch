import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import matplotlib.pyplot as plt
from maddpg.src.utils.loadSaveModel import loadFromPickle
import numpy as np 
sheepMaxSpeedOriginal = 1.3

def main():
    resultPath = os.path.join(dirName, '..', 'evalResults')
    allResultLoc = os.path.join(resultPath, 'evalAll.pkl')
    resultDF = loadFromPickle(allResultLoc)

    independentVariables = dict()
    independentVariables['num_predators'] = [3, 4, 5, 6]
    independentVariables['speed'] = [1]
    independentVariables['cost'] = [0, 0.01, 0.02, 0.03]
    independentVariables['selfish'] = [0, 1, 10000]

    figure = plt.figure(figsize=(5, 15))
    plotCounter = 1

    numRows = len(independentVariables['selfish']) 
    numColumns = len(independentVariables['speed'])

    for key, outmostSubDf in resultDF.groupby('selfish'): 
        outmostSubDf.index = outmostSubDf.index.droplevel('selfish') 
        for keyCol, outterSubDf in outmostSubDf.groupby('speed'):
            outterSubDf.index = outterSubDf.index.droplevel('speed')
            axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
            for keyRow, innerSubDf in outterSubDf.groupby('cost'):
                innerSubDf.index = innerSubDf.index.droplevel('cost')
                plt.ylim([0, 25])

                innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('Prey Speed = ' + str(np.round(keyCol* sheepMaxSpeedOriginal, 1)) + 'x')
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('Predators Selfish Level = ' + str(key))
                axForDraw.set_xlabel('Number of Predators')

            plotCounter += 1
            plt.xticks(independentVariables['num_predators'])
            plt.legend(title='action cost', title_fontsize = 8, prop={'size': 8})

    figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate predatorSelfishness/ preySpeed/ actionCost')
    plt.savefig(os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_killNum_allcond_regroup'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

