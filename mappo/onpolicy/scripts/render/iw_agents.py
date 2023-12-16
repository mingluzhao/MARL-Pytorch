import numpy as np
import copy
import scipy.stats as ss
import itertools as it
import pandas as pd


class ChaseAgent:
    def __init__(self, all_args, agent_id, agent_trainer):
        self.agent_id = agent_id
        self.agent_trainer = agent_trainer

        self.rnn_state = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        self.mask = np.ones((1, 1), dtype=np.float32)
        self.action_dim = all_args.action_dim

        self.num_agents = all_args.num_agents
        self.num_prey = all_args.num_good_agents
        self.num_pred = all_args.num_adversaries
        self.pred_ids = list(range(self.num_pred))
        self.prey_ids = list(range(self.num_pred, self.num_pred + self.num_prey))
        self.setup_inference()
        self.softenPrior = SoftDistribution(softParameter = 1)
        self.noise = 1/np.sqrt(5)

        self.visible_predator_id = None if agent_id in self.pred_ids else self.pred_ids
        self.visible_prey_id = None if agent_id in self.pred_ids else [self.agent_id]

        # for intention
        self.infer_time_step = 0
        self.last_world = None
        self.lastAction = None

    def set_visible_id(self, id_we):
        # e.g. give [0, 1, 2, 4]
        visible_predator_id = []
        visible_prey_id = []
        for id in id_we:
            if id in self.pred_ids:
                visible_predator_id.append(id)
            else:
                visible_prey_id.append(id)
        self.visible_predator_id = visible_predator_id
        self.visible_prey_id = visible_prey_id

    def setup_inference(self):
        self.intentionSpace = tuple(it.product(self.prey_ids, [tuple(self.pred_ids)]))
        self.intentionPrior = {tuple(intention): 1 / len(self.intentionSpace) for intention in self.intentionSpace}
        self.jointHypothesisSpace = pd.MultiIndex.from_product([self.intentionSpace], names=['intention'])
        self.concernedHypothesisVariable = ['intention']
        self.formerIntentionPriors = [self.intentionPrior]

    def infer_one_step(self, intentionPrior, likelihood):
        if self.num_prey == 1:
            return intentionPrior

        jointHypothesisDf = pd.DataFrame(index=self.jointHypothesisSpace)
        jointHypothesisDf['likelihood'] = likelihood
        marginalLikelihood = jointHypothesisDf.groupby(self.concernedHypothesisVariable).sum()
        oneStepLikelihood = marginalLikelihood['likelihood'].to_dict()

        softPrior = self.softenPrior(intentionPrior)
        unnomalizedPosterior = {key: np.exp(np.log(softPrior[key] + 1e-4) + np.log(oneStepLikelihood[key])) for key
                                in list(intentionPrior.keys())}
        normalizedProbabilities = np.array(list(unnomalizedPosterior.values())) / np.sum(
            list(unnomalizedPosterior.values()))
        normalizedPosterior = dict(zip(list(unnomalizedPosterior.keys()), normalizedProbabilities))
        return normalizedPosterior

    # def update_intention(self, state):
    #     adjustedIntentionPrior = self.intentionPrior.copy()  # this one used
    #
    #     if self.infer_time_step == 0:
    #         intentionPosterior = adjustedIntentionPrior.copy()
    #     else:
    #         perceivedAction = self.perceive_iw_noisy_action(self.lastAction)
    #         intentionPosterior = self.infer_one_step(adjustedIntentionPrior, self.lastState, perceivedAction)
    #
    #     intention = sampleFromDistribution(intentionPosterior)
    #
    #     self.lastState = state.copy()
    #     self.intentionPrior = intentionPosterior.copy()
    #     self.formerIntentionPriors.append(self.intentionPrior.copy())
    #     self.infer_time_step = self.infer_time_step + 1
    #     return intention

    def resetObjects(self):
        intentionPrior = {tuple(intention): 1 / len(self.intentionSpace) for intention in self.intentionSpace}
        self.infer_time_step = 0
        self.lastState = None
        self.lastAction = None
        self.intentionPrior = intentionPrior
        self.formerIntentionPriors = [intentionPrior]

    def getIntentionDistributions(self):
        return self.formerIntentionPriors

    def recordActionForUpdateIntention(self):
        return self.lastAction

    def sample_noisy_action(self, agent_action):
        # noisy_action = np.random.multivariate_normal(agent_action, np.diag([self.noise**2] * len(agent_action)))
        # TODO: mappo now discrete, action = 5d number
        noisy_action = np.random.multivariate_normal(agent_action, np.diag([self.noise**2] * len(agent_action)))

        return noisy_action

    def perceive_iw_noisy_action(self, all_actions):
        return np.array([self.sample_noisy_action(agent_action) for agent_action in all_actions])

    def observe(self, world):
        agent = world.agents[self.agent_id]
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        other_pos = []
        other_vel = []
        visible_agents_id = self.visible_predator_id + self.visible_prey_id
        for other_agent_id in visible_agents_id:
            if other_agent_id == self.agent_id: continue
            other = world.agents[other_agent_id]
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        ag_health = [world.agents[preyid].health for preyid in self.visible_prey_id]
        agent_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + [ag_health])
        agent_obs = agent_obs.reshape(1, -1)
        return agent_obs

    def act(self, agent_obs, save_rnn = False):
        self.agent_trainer.prep_rollout()
        action, rnn_state = self.agent_trainer.policy.act(np.array(list(agent_obs)), self.rnn_state, self.mask, deterministic=True)
        action = action.detach().cpu().numpy()
        agent_action = np.squeeze(np.eye(self.action_dim)[action], 1)[0]
        if save_rnn:
            self.rnn_state = rnn_state.detach().cpu().numpy()

        return agent_action

    def chooseCommittedAction(self, jointActionDistribution, weIds):
        jointAction = tuple([tuple(distribution.rvs()) for distribution in jointActionDistribution])
        action = tuple(np.array(jointAction)[list(weIds).index(self.agent_id)])
        return action

    def chooseUncommittedAction(self, distribution):
        return sampleFromDistribution(distribution)

    # def sampleIndividualActionGivenIntention(self, intention, jointActionDist):
    #     goalId, weIds = intention
    #     if self.agent_id not in list(weIds):
    #         individualAction = self.chooseUncommittedAction(jointActionDist)
    #     else:
    #         individualAction = self.chooseCommittedAction(jointActionDist, weIds)
    #
    #     return individualAction
    #
    # def act_with_intention(self, jointActionDist):
    #     intention = self.update_intention(state)
    #     individualAction = self.sampleIndividualActionGivenIntention(intention, jointActionDist)
    #     return individualAction

    def record_action(self, action):
        self.lastAction = action


class SoftDistribution:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, distribution):
        hypotheses = list(distribution.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter)for probability in list(distribution.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenDistribution = dict(zip(hypotheses, softenNormalizedProbabilities))
        return softenDistribution

def sampleFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis


class IW:
    def __init__(self, all_args, trainer):
        self.rationalityBeta = 1.0
        self.agents = []
        self.num_agents = all_args.num_agents
        self.num_prey = all_args.num_good_agents
        self.num_pred = all_args.num_adversaries
        self.pred_ids = list(range(self.num_pred))
        self.prey_ids = list(range(self.num_pred, self.num_pred + self.num_prey))
        self.verbose = all_args.render_verbose

        for agent_id in range(self.num_agents):
            agent_trainer = trainer[agent_id]
            agent = ChaseAgent(all_args, agent_id, agent_trainer)
            self.agents.append(agent)

    def reset(self):
        for agent in self.agents:
            agent.resetObjects()

    def act_raw(self, world):
        all_actions = []
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            agent_obs = agent.observe(world)
            agent_action = agent.act(agent_obs)
            all_actions.append(agent_action)
        return all_actions

    def composeCentralPolicy(self, world, ids_we, current_id = None, cov = None):
        actionDimReshaped = 2
        # covForPlanning = [cov ** 2 for _ in range(actionDimReshaped)]
        central_policy_dist = []
        for agent_id in ids_we:
            agent = self.agents[agent_id]
            agent.set_visible_id(ids_we)
            agent_obs = agent.observe(world)
            if current_id and current_id == agent_id:
                agent_action = agent.act(agent_obs, save_rnn = True)
            else:
                agent_action = agent.act(agent_obs)
            agent_action_noisy = ss.multivariate_normal(agent_action, np.diag([cov ** 2] * len(agent_action)))
            central_policy_dist.append(agent_action_noisy)

        return central_policy_dist

    def policyForCommittedAgentInPlanning(self, world, goalId, weIds, agent_id):
        IdsRelative = list(np.array([goalId]).flatten()) + list(weIds)
        ids_we = np.array(np.sort(IdsRelative))
        actionDistribution = self.composeCentralPolicy(world, ids_we, current_id = agent_id, cov = 0.03/np.sqrt(5)) # 0.03
        return actionDistribution

    def policyForCommittedAgentsInInference(self, world, goalId, weIds):
        IdsRelative = list(np.array([goalId]).flatten()) + list(weIds)
        ids_we = np.array(np.sort(IdsRelative))
        actionDistribution = self.composeCentralPolicy(world, ids_we, current_id = None, cov = 1/np.sqrt(5)) # cov = 1
        return actionDistribution

    def policy_random(self):
        actionSpace = [(5, 0), (3.5, 3.5), (0, 5), (-3.5, 3.5), (-5, 0), (-3.5, -3.5), (0, -5), (3.5, -3.5), (0, 0)]
        actionDist = {action: 1 / len(actionSpace) for action in actionSpace}
        return actionDist

    def get_committed_policy_lik(self, intention, world, perceived_action):
        goalId, weIds = intention
        committedAgentIds = [Id for Id in list(weIds) if Id in self.pred_ids]
        if len(committedAgentIds) == 0:
            committedAgentsPolicyLikelihood = 1
        else:
            jointActionDistribution = self.policyForCommittedAgentsInInference(world, goalId, weIds)
            jointAction = np.array(perceived_action)[list(committedAgentIds)]
            pdfs = [individualDistribution.pdf(action) for individualDistribution, action in zip(jointActionDistribution, jointAction)]
            committedAgentsPolicyLikelihood = np.power(np.product(pdfs), self.rationalityBeta)
        return committedAgentsPolicyLikelihood

    def get_uncommitted_policy_lik(self, intention, perceived_action):
        goalId, weIds = intention
        uncommittedAgentIds = [Id for Id in self.pred_ids if (Id not in list(weIds))]
        if len(uncommittedAgentIds) == 0:
            return 1
        else:
            uncommittedActionDistributions = [self.policy_random() for _ in range(len(uncommittedAgentIds))]
            uncommittedAgentsPolicyLikelihood = np.product([actionDistribution[tuple(perceived_action[Id])] for actionDistribution, Id in zip(uncommittedActionDistributions, uncommittedAgentIds)])
            return uncommittedAgentsPolicyLikelihood

    def get_joint_lik(self, intention, world, perceived_action):
        return self.get_committed_policy_lik(intention, world, perceived_action) * self.get_uncommitted_policy_lik(intention, perceived_action)

    def act(self, world):
        agents_actions = []
        for agent_id in self.pred_ids:
            agent = self.agents[agent_id]

            # infer intention distribution by previous info, generate intention
            intentionPrior = agent.intentionPrior.copy()
            if agent.infer_time_step == 0:
                intentionPosterior = intentionPrior.copy()
            else:
                perceivedAction = agent.perceive_iw_noisy_action(agent.lastAction)
                likelihood = [self.get_joint_lik(intention, agent.last_world, perceivedAction) for intention in agent.intentionSpace]
                intentionPosterior = agent.infer_one_step(intentionPrior, likelihood)

            intention = sampleFromDistribution(intentionPosterior)
            if self.verbose:
                print(f"Agent{agent_id} intention dist {intentionPosterior}, sampled {intention}")

            agent.last_world = copy.deepcopy(world)
            agent.intentionPrior = intentionPosterior.copy()
            agent.formerIntentionPriors.append(agent.intentionPrior.copy())
            agent.infer_time_step = agent.infer_time_step + 1

            # generate actions
            goalId, weIds = intention
            if agent_id not in list(weIds):
                jointActionDist = self.policy_random()
                agent_action = agent.chooseUncommittedAction(jointActionDist)
            else:
                jointActionDist = self.policyForCommittedAgentInPlanning(world, goalId, weIds, agent_id)
                agent_action = agent.chooseCommittedAction(jointActionDist, weIds)

            agents_actions.append(agent_action)

        for agent_id in self.prey_ids:
            agent = self.agents[agent_id]
            agent_obs = agent.observe(world)
            agent_action = agent.act(agent_obs, save_rnn = True)
            agents_actions.append(agent_action)

        for agent in self.agents:
            agent.record_action(agents_actions)

        return agents_actions


