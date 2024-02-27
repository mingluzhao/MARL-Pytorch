from psychopy import visual, core, event
import numpy as np
import os
import copy
import pickle

def saveToPickle(data, path):
    with open(path, "wb") as pklFile:
        pickle.dump(data, pklFile)

def loadFromPickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def drawCircle(wPtr, pos, size):
    circle = visual.Circle(win=wPtr, lineColor='grey', colorSpace='rgb255', pos=pos, size=size)
    return circle


def showText(wPtr, textHeight, text, position):
    introText = visual.TextStim(wPtr, height=textHeight, font='Times sheep Roman', text=text, pos=position)
    return introText


def expandCoordination(position, expandRatio):
    return [[i[j] * expandRatio for j in range(len(position[0]))] for i in position]


def main():
    textHeight = 24
    expandRatio = 300 * 0.95
    wolfRadius = 0.065
    sheepRadius = 0.065
    objectSize = wolfRadius * 2 * expandRatio
    targetSize = sheepRadius * 2 * expandRatio
    targetColorsOrig = ['orange', 'purple']
    waitTime = 0.05

    wPtr = visual.Window(size=[610, 610], units='pix', fullscr=False)
    myMouse = event.Mouse(visible=True, win=wPtr)
    introText = showText(wPtr, textHeight, 'Press ENTER to start', (0, 250))
    introText.autoDraw = True
    wPtr.flip()

    while 'return' not in event.getKeys():
        pass
    introText.autoDraw = False

    dirName = os.path.dirname(__file__)
    fileFolder = os.path.join(dirName, '..', 'results', 'testResult')
    prefixList = [os.path.splitext(j)[0] for j in os.listdir(fileFolder) if os.path.splitext(j)[1] == '.pickle']

    for prefix in prefixList:
        numWolves = 3
        fileName = os.path.join(dirName, '..', 'results', 'testResult', prefix + '.pickle')
        allRawData = loadFromPickle(fileName)

        sheepPickleData = []
        allTrialList = list(range(len(allRawData)))

        for i in allTrialList:
            wPtr.flip()
            oneTrialRawData = copy.deepcopy(allRawData[i])
            trajectory = oneTrialRawData['trajectory']
            condition = oneTrialRawData['condition']
            numSheep = condition['sheepNums']
            blockSize = condition['blockSize']
            obstacleSizeExpanded = blockSize * 2 * expandRatio

            if blockSize <= 0:
                numBlocks = 0
            else:
                numBlocks = 2

            if 'targetColorIndex' in condition and condition['targetColorIndex'] != 'None':
                targetColorIndex = condition['targetColorIndex']
                targetColor = [targetColorsOrig[index] for index in targetColorIndex]
                numSheep = len(targetColorIndex)
                targetNumList = [numSheep]
                if i + 1 not in [42]:
                    continue
            sheepPickleData.append(oneTrialRawData)

            targetPoses = [[], [], [], []]
            playerPoses = [[], [], []]
            blockPoses = [[], []]

            for timeStep in trajectory:
                state = timeStep[0]
                for wolfIndex in range(0, numWolves):
                    playerPoses[wolfIndex].append(state[wolfIndex][0:2])
                for sheepIndex in range(numWolves, numWolves + numSheep):
                    targetPoses[sheepIndex - numWolves].append(state[sheepIndex][0:2])
                for blockIndex in range(numWolves + numSheep, numWolves + numSheep + numBlocks):
                    blockPoses[blockIndex - numWolves - numSheep].append(state[blockIndex][0:2])

            targetPos1, targetPos2, targetPos3, targetPos4 = targetPoses
            playerPos1, playerPos2, playerPos3 = playerPoses
            blockPos1, blockPos2 = blockPoses

            drawBlockCircleFun = lambda pos: drawCircle(wPtr, pos[0], obstacleSizeExpanded)

            if numBlocks == 2:
                blockPos1 = expandCoordination(blockPos1, expandRatio)
                block1Traj = drawBlockCircleFun(blockPos1)
                block1Traj.setFillColor('white')
                block1Traj.autoDraw = True
                blockPos2 = expandCoordination(blockPos2, expandRatio)
                block2Traj = drawBlockCircleFun(blockPos2)
                block2Traj.setFillColor('white')
                block2Traj.autoDraw = True

            drawTargetCircleFun = lambda pos: drawCircle(wPtr, pos[0], targetSize)
            targetPos1 = expandCoordination(targetPos1, expandRatio)
            target1Traj = drawTargetCircleFun(targetPos1)
            if targetNumList[0] == 2:
                targetPos2 = expandCoordination(targetPos2, expandRatio)
                target2Traj = drawTargetCircleFun(targetPos2)

            expandFun = lambda pos: expandCoordination(pos, expandRatio)
            playerPos1, playerPos2, playerPos3 = expandFun(playerPos1), expandFun(playerPos2), expandFun(playerPos3)
            drawPlayerCircleFun = lambda pos: drawCircle(wPtr, pos[0], objectSize)
            player1Traj, player2Traj, player3Traj = drawPlayerCircleFun(playerPos1), drawPlayerCircleFun(
                playerPos2), drawPlayerCircleFun(playerPos3)

            player1Traj.setFillColor('red')
            player2Traj.setFillColor('blue')
            player3Traj.setFillColor('green')

            target1Traj.setFillColor(targetColor[0])
            if targetNumList[0] == 2:
                target2Traj.setFillColor(targetColor[1])

            player1Traj.autoDraw = True
            player2Traj.autoDraw = True
            player3Traj.autoDraw = True
            target1Traj.autoDraw = True
            if targetNumList[0] == 2:
                target2Traj.autoDraw = True

            stepCount = 0
            for x, y, z, a in zip(playerPos1, playerPos2, playerPos3, targetPos1):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    break

            if numBlocks == 2:
                block1Traj.autoDraw = False
                block2Traj.autoDraw = False
            player1Traj.autoDraw = False
            player2Traj.autoDraw = False
            player3Traj.autoDraw = False
            target1Traj.autoDraw = False
            if targetNumList[0] == 2:
                target2Traj.autoDraw = False

        saveToPickle(sheepPickleData, os.path.join(dirName, '..', 'results', prefix + '.pickle'))
        wPtr.flip()
        event.waitKeys()
    wPtr.close()





if __name__ == "__main__":
    main()
