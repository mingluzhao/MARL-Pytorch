from multiprocessing import Process, Pipe
import numpy as np


class CustomEnv:
    def __init__(self, reset, observe, transit, rewardFunc, isTerminal):
        self.reset = reset
        self.observe = observe
        self.transit = transit
        self.rewardFunc = rewardFunc
        self.isTerminal = isTerminal

    def step(self, state, action):
        nextState = self.transit(state, action)
        nextObs = self.observe(nextState)
        rewards = self.rewardFunc(state, action, nextState)
        dones = self.isTerminal(state)
        infos = None

        return nextState, nextObs, rewards, dones, infos

    def reset(self):
        return self.reset()

    def observe(self, state):
        obs = self.observe(state)
        obs = np.array([obs])
        return obs


# Worker function for multiprocessing
def worker(remote, parent_remote, reset, observe, transit, rewardFunc, isTerminal):
    parent_remote.close()
    env = CustomEnv(reset, observe, transit, rewardFunc, isTerminal)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            state, action = data
            nextState, nextObs, rewards, dones, infos = env.step(state, action)
            remote.send((nextState, nextObs, rewards, dones, infos))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'observe':
            ob = env.observe(data)  # data here is the state for observation
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


# Class to manage multiple environments
class ParallelEnv:
    def __init__(self, num_envs, reset, observe, transit, rewardFunc, isTerminal):
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, reset, observe, transit, rewardFunc, isTerminal))
                   for work_remote, remote in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, states, actions):
        for remote, state, action in zip(self.remotes, states, actions):
            remote.send(('step', [state, action]))
        results = [remote.recv() for remote in self.remotes]
        nextState, nextObs, rewards, dones, infos = zip(*results)
        return np.stack(nextState), np.stack(nextObs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def observe(self, states):
        for remote, state in zip(self.remotes, states):
            remote.send(('observe', state))
        observations = [remote.recv() for remote in self.remotes]
        return np.stack(observations)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


class EvalVecEnv:
    def __init__(self, reset, observe, transit, rewardFunc, isTerminal):
        self.envs = [CustomEnv(reset, observe, transit, rewardFunc, isTerminal)]

    def step(self, states, actions):
        results = [env.step(state, action) for env, state, action in zip(self.envs, states, actions)]
        nextState, nextObs, rewards, dones, infos = zip(*results)
        return np.array(nextState), np.array(nextObs), np.array(rewards), np.array(dones), infos

    def reset(self):
        return np.array([env.reset() for env in self.envs])

    def observe(self, states):
        return np.array([env.observe(state) for env, state in zip(self.envs, states)])

