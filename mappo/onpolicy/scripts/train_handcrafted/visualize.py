
import numpy as np

import pandas as pd
import sys
import os
from pathlib import Path
import torch
import json
import pickle
from collections import OrderedDict
import glob
from psychopy import visual, core
import cv2
from PIL import Image

from onpolicy.envs.hunting_environment.chasingEnv.multiAgentEnv import ObserveWithCaughtHistory,\
    getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object

def drawCircle(wPtr, pos, size, color):
    circle = visual.Circle(win=wPtr, lineColor=color, fillColor=color, pos=pos, size=size)
    return circle

def expandCoordination(position, expandRatio):
    return [[p[j] * expandRatio for j in range(len(p))] for p in position]

def visualize(numSheep, trainSheepSpeed, evalSheepSpeed, individualRewardWolf, continuous):
    discrete_action = not continuous
    numWolves = 3
    numBlocks = 0
    numAgents = numWolves + numSheep
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))

    if continuous:
        cont = "continuous"
    else:
        cont = "discrete"

    pattern = f'../results/trajectories/{cont}/{numWolves}pred_{numSheep}prey-{numBlocks}block-sheepspeed{trainSheepSpeed}-indivd{individualRewardWolf}-discrete{discrete_action}-seed*-evalspeed{evalSheepSpeed}.pkl'
    exp_name = glob.glob(pattern)[0]
    start_index = exp_name.find(f"{numWolves}pred")
    foldername = exp_name[start_index:]
    print(exp_name)
    print(foldername)


    # save_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + f"/results/trajectories/{cont}")
    # traj_loc = os.path.join(save_dir, f'{foldername}-evalspeed{evalSheepSpeed}.pkl')
    traj_list = loadFromPickle(exp_name)

# --------------
    wolves_trajectories = [[] for _ in range(numWolves)]
    sheep_trajectories = [[] for _ in range(numSheep)]

    for traj in traj_list:
        for timestep in traj:
            state = timestep[0][0]
            for wolf_id in wolvesID:
                wolves_trajectories[wolf_id].append(state[wolf_id][0:2])

            for i, sheep_id in enumerate(sheepsID):
                sheep_trajectories[i].append(state[sheep_id][0:2])


    expandRatio = 300 * 0.95
    wolfRadius = 0.065
    sheepRadius = 0.065
    objectSize = wolfRadius * 2 * expandRatio
    targetSize = sheepRadius * 2 * expandRatio

    wPtr = visual.Window(size=[610, 610], units='pix', fullscr=False, winType='pyglet', allowGUI=False)
    waitTime = 0.05


    wolves_trajectory = [expandCoordination(traj, expandRatio) for traj in wolves_trajectories]
    sheep_trajectory = [expandCoordination(traj, expandRatio) for traj in sheep_trajectories]

    wolves_circles = [drawCircle(wPtr, traj[0], objectSize, 'red') for traj in wolves_trajectory]
    # sheep_circles = [drawCircle(wPtr, traj[0], objectSize, 'green') for traj in sheep_trajectory]
    sheep_circles = [drawCircle(wPtr, sheep_trajectory[0][0], objectSize, 'green') , drawCircle(wPtr, sheep_trajectory[1][0], objectSize, 'blue')]

    for circle in wolves_circles + sheep_circles:
        circle.autoDraw = True

    frames = []

    for step in range(len(wolves_trajectory[0])):
        for i, wolf_circle in enumerate(wolves_circles):
            wolf_circle.setPos(wolves_trajectory[i][step])
        for i, sheep_circle in enumerate(sheep_circles):
            sheep_circle.setPos(sheep_trajectory[i][step])

        wPtr.flip()
        core.wait(waitTime)

        # Capture the frame
        frame = np.array(wPtr.getMovieFrame())
        frames.append(frame)
        wPtr.movieFrames.append(Image.fromarray(frame))

    for circle in wolves_circles + sheep_circles:
        circle.autoDraw = False

    wPtr.close()

    # Create and save the video
    height, width, layers = frames[0].shape
    size = (width, height)

    dirName = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results/eval_videos")
    if not dirName.exists():
        os.makedirs(str(dirName))

    out = cv2.VideoWriter(f'{dirName}/{foldername}-evalspeed{evalSheepSpeed}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, size)

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def main():
    numSheep = 2
    trainSheepSpeed = 1.1
    evalSheepSpeed = 1.1
    individualRewardWolf = 0
    continuous = True
    visualize(numSheep, trainSheepSpeed, evalSheepSpeed, individualRewardWolf, continuous)



if __name__ == "__main__":
    main()
