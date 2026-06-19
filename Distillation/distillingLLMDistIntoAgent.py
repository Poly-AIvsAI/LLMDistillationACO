#From https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/main.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
import matplotlib.pyplot as plt
import numpy as np
from CybORG import CybORG
import inspect
from CybORG.Agents import B_lineAgent, GreenAgent, BlueMonitorAgent, RedMeanderAgent, OrigB_lineAgent, B_lineAgent4,\
    B_lineAgent5,B_lineAgent6,B_lineAgent7,B_lineAgent8,B_lineAgent9,B_lineAgent10,B_lineAgent11,B_lineAgent12, B_lineAgent13, B_lineAgent13Deterministic
from CybORG.Agents.Wrappers import ChallengeWrapper, OpenAIGymWrapper, EnumActionWrapper, FixedFlatWrapper
from pprint import pprint
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
from CybORG.Shared.Actions.AbstractActions.Impact import Impact
import matplotlib.pyplot as plt
import time
import datetime
from CybORG.Shared.Actions.AbstractActions.Impact import Impact #For checking if action is Impact (for terminating condition)
from CybORG.Shared.Actions import BlockTraffic, AllowTraffic, IsolateHost, UnIsolateHost
from pathlib import Path
import json
import sys
import os
import sqlite3
import math
import optuna
import yaml
from collections import deque, defaultdict
# import #wandb

#In case calling from different location, need to add to path (or can add to PYTHONPATH or install as python module)
sys.path.append(str(Path(__file__).parent.parent))
from customWrapper import CustomBlueWrapper


#wandbLogDict={}
groupName="LLMStandardIntegration_AuxLossAndActionMask"

def getLRScheduler(optimizer,initialLR,minLR,decayRate):
    lrLambda = lambda epoch: max(decayRate**epoch, minLR/initialLR)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrLambda)

class ActorNetwork(nn.Module):
    def __init__(self, actionDim: int, stateDim: int, learnRate, hiddenDimVector: list[int]=[256,128,64]):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(stateDim, hiddenDimVector[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDimVector[0], hiddenDimVector[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDimVector[1], hiddenDimVector[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDimVector[2], actionDim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learnRate)
        self.scheduler=getLRScheduler(optimizer=self.optimizer,initialLR=learnRate,minLR=1e-5,decayRate=0.98)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        #Using Categorical to allow easy sampling and log_prob calculation (and because recommended by Phil)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        #self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, stateDim: int, learnRate, hiddenDimVector: list[int]=[256,128,64]):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(stateDim, hiddenDimVector[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDimVector[0], hiddenDimVector[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDimVector[1], hiddenDimVector[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDimVector[2], 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learnRate)
        self.scheduler=getLRScheduler(optimizer=self.optimizer,initialLR=learnRate,minLR=1e-5,decayRate=0.95)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        #self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, actionSize, stateSize, gamma=0.99, policyLR=0.0003, criticLR=0.0003, gaeLambda=0.95,
            policyClip=0.1, epochs=10, entropyCoef=0.001, entropyDecay=0.99, policyGradClip=0.5, criticGradClip=0.1):
        self.gamma = gamma
        self.policyClip = policyClip
        self.epochs = epochs
        self.gaeLambda = gaeLambda

        self.actor = ActorNetwork(actionSize, stateSize, policyLR)
        self.critic = CriticNetwork(stateSize, criticLR)

        self.entropyCoef=entropyCoef
        self.entropyDecay=entropyDecay
        self.policyGradClip=policyGradClip
        # self.criticGradClip=criticGradClip
        #STARTING AT 1 FOR OPTIMIZED AUX LOSS
        self.criticGradClip=1

        self.entropyWeight=entropyCoef
        self.entropyDecay=entropyDecay
        self.pretrainedScale=0
        self.trgIntervalCounter=0

        self.criticWeight=0.9
        self.transitioned=False #For when transition from teacher-guided to independent
    #Explained Variance quanitfies how well critic explains returns
    def calcExplainedVariance(self, yTrue, yPred):
        varY = np.var(yTrue)
        return 1 - np.var(yTrue - yPred) / (varY + 1e-8)

    #Save the model. Save using state_dict is the recommended approach per https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def savePolicyModel(self, path):
        torch.save(self.actor.state_dict(), path)
    
    #Load the model. Since using state_dict, need to instantiate the model first and then load the learned parameters
    def loadPolicyModel(self, path):
        self.actor.load_state_dict(torch.load(path))

    def saveCriticModel(self,path):
        torch.save(self.critic.state_dict(), path)
    
    def loadCriticModel(self,path):
        self.critic.load_state_dict(torch.load(path))
        
    def getAction(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        return dist,value
 
    #Calculating advantage vector using Phil's alg (ref: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py)
    def computeAdvantage(self,rewardArr, donesArr, criticVals):

        #GAE ripped off from stablebaseline3 (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
        advantage = np.zeros(len(rewardArr), dtype=np.float32)
        last_gae_lam = 0  # Tracks accumulated advantage
        
        #Go in reverse to do it in single pass
        for t in reversed(range(len(rewardArr) - 1)): 
            next_non_terminal = 1 - int(donesArr[t])  # Next value to 0 if done=True, 1 otherwise

            #TD(1) getting difference between actual reward & next criticVal*discount and critic's estimate (criticVals[t])
            delta = rewardArr[t] + self.gamma * criticVals[t + 1] * next_non_terminal - criticVals[t]

            #Add discounted past (future, but past since going backwards) to the discount
            last_gae_lam = delta + self.gamma * self.gaeLambda * next_non_terminal * last_gae_lam
            
            advantage[t] = last_gae_lam  # Store computed advantage

        return torch.tensor(advantage).to(self.actor.device)

    def learn(self, trainingData, recommendedDists=np.array([]), probabilityModifier=None):
        criticLossArr=[]
        actorLossArr=[]
        criticValsArr=[]
        returnsArr=[]
        criticPredsArr=[]
        entropyArr=[]
        numLowClips=0
        numHighClips=0


        for epoch in range(self.epochs):
            #Unpack the training data
            obsArr, actionArr, oldProbArr, criticVals,\
            rewardArr, donesArr, batches = trainingData
            #Compute the advantage for how favorable each action is 
            #Can see that advantage stays the same (since calculated with sampled data - this is by design)
            advantage=self.computeAdvantage(rewardArr, donesArr, criticVals)
            #Normalize the advantage to ensure its standard deviation=1 and mean=0
            # advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
            criticVals = torch.tensor(criticVals).to(self.critic.device)
            #Iterate through each batch            
            for batch in batches:
                states = torch.tensor(obsArr[batch], dtype=torch.float).to(self.actor.device)
                oldProbs = torch.tensor(oldProbArr[batch]).to(self.actor.device)
                actions = torch.tensor(actionArr[batch]).to(self.actor.device)
                if recommendedDists:
                    recommendedDistsBatch = torch.stack(recommendedDists)[batch]  # Shape: [batch_size, num_actions]
                    recommendedDistsBatch = recommendedDistsBatch.to(self.actor.device).float()
                else:
                    recommendedDistsBatch=None
                #Get the current distribution and critic value of the state
                dist=self.actor(states)
                criticVal = self.critic(states)
                criticVal = torch.squeeze(criticVal)

                # Compute new log probabilities
                newProbs = dist.log_prob(actions)
                probRatio = newProbs.exp() / oldProbs.exp()

                #OR CAN DO (ref https://www.youtube.com/watch?v=xHf8oKd7cgU):
                #prob_ratio = (new_probs - old_probs).exp()

                #Multiplying probability ratio by advantage (to get critic's input)
                weightedProbs = advantage[batch] * probRatio
                #Clamping the ratio to prevent too large of an update
                weightedClippedProbs = torch.clamp(probRatio, 1-self.policyClip,
                        1+self.policyClip)*advantage[batch]
                
                #Taking negative of actor loss since want to maximize reward
                actorLoss = -torch.min(weightedProbs, weightedClippedProbs).mean()

                #Computing the "real" value of state (to calculate loss for back prop)
                returns = advantage[batch] + criticVals[batch]

                #Decided to just clip critic at gradient instead of in loss calculation
                # clippedCriticVal = criticVals[batch] + torch.clamp(criticVal - criticVals[batch], -0.2, 0.2)
                criticLoss = (returns-criticVal).pow(2)
                criticLoss = criticLoss.mean()
                # print("CRITIC LOSS: ", criticLoss)

                #Calculated the entropy in the new distribution (stable baseline implementation)
                entropy = dist.entropy()
                entropyLoss = -torch.mean(entropy)

                if recommendedDists:
                    #Account for pretrained loss!
                    #Ensure that pretrained loss is scaled proportional to the actor loss.
                    # #PRETRAINED LOSS WITH SINGLE ACTIONS
                    # pretrainedProbs=dist.log_prob(recommendedDistsBatch)
                    # pretrainedLoss=-pretrainedProbs.mean()

                    #PRETRAINED LOSS WITH ENTIRE DIST
                    agentProbs=dist.probs.clamp(min=1e-8) #Clamp to avoid log(0)
                    llmProbs=recommendedDistsBatch.clamp(min=1e-8)
                    llmProbs=llmProbs.squeeze(1) #Get rid of singleton dimension

                    pretrainedLoss=torch.nn.functional.kl_div(
                        input=agentProbs.log(),
                        target=llmProbs,
                        reduction='batchmean'
                    )
                   
                    #Ensure that pretrained loss doesn't exceed actor loss
                    totalLoss = (self.pretrainedScale)*actorLoss + (1-self.pretrainedScale)*pretrainedLoss + self.criticWeight*criticLoss\
                          + (self.pretrainedScale)*self.entropyWeight*entropyLoss
                else:
                    totalLoss=actorLoss+0.5*criticLoss#+ self.entropyWeight*entropyLoss

                #Pytorch by default accumulates gradients, so clear to ensure no interference
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                
                #Since total_loss was calculated using actor and critic, will automatically calculate gradients for both
                totalLoss.backward()
                #Move in single step of LR towards opposite direction of gradient (to minimize loss)

                #Clip gradients to ensure don't exceed 0.5 for actor and 0.1 for critic
                #Clipping critic more because noticed its loss would be quite large (show this in thesis)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policyGradClip)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.criticGradClip)
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                criticLossArr.append(criticLoss)
                actorLossArr.append(actorLoss)
                criticValsArr.append(criticVal.mean().item())
                entropyArr.append(entropy.mean().item())
                #For calculating explained variance (how good the critic is at predicting)
                returnsArr.append(criticVal.detach().cpu().numpy())
                criticPredsArr.append(returns.detach().cpu().numpy())


        #----------------For optimized (already optimized so slower and more stable!)
        # Aux loss decay
        # Gradual decrease after 10
        # decreaseAfter=1
        # if self.pretrainedScale < 1 and self.trgIntervalCounter >= decreaseAfter:
        #     self.pretrainedScale=min(1, self.pretrainedScale+0.1)
        #     self.entropyWeight+=0.0002
        #     # print("DECAYING AUX LOSS")
        # elif self.entropyWeight > 0.005 and self.trgIntervalCounter>= decreaseAfter:
        #         self.entropyWeight=max(0.005,self.entropyWeight-0.0001)
        # elif self.trgIntervalCounter>= decreaseAfter:
        #         self.entropyCoef*=self.entropyDecay

        # #Action mask decay
        # # #Gradually decay by 0.1 as well!
        # if probabilityModifier and self.trgIntervalCounter>=1:
        #     probabilityModifier.impactScale=min(1, probabilityModifier.impactScale+0.2)
        #     # print("DECAYING ACTION MASK")

        # #STANDARD
            # self.entropyWeight+=0.0001
        # elif self.trgIntervalCounter>=30:
        #     # if self.entropyCoef <= 0.001:
        #     #     self.entropyWeight=max(0.001, self.entropyWeight-0.0001)

            
            # #Do transition when pretrained scale is 0.8
            # if not self.transitioned and self.pretrainedScale >=0.8:
            #     self.transitioned=True
            #     for param_group in self.critic.optimizer.param_groups:
            #         param_group['lr'] *= 0.125 #Multiplying by 0.125 since started off as x2
            #     for param_group in self.actor.optimizer.param_groups:
            #         param_group['lr'] *= 0.25
            #     self.criticWeight=0.5
            #     self.policyClip=0.1

        # # #Gradually decay by 0.1 as well!
        # if probabilityModifier and self.trgIntervalCounter>=1:
        #     probabilityModifier.impactScale=min(1, probabilityModifier.impactScale+0.2)

        # #------------For standard (bit quicker to train)
        # # #Gradual decrease after 1
        # if self.trgIntervalCounter < 1:
        #     pass
        # elif self.pretrainedScale < 1:
        #     self.pretrainedScale=min(1, self.pretrainedScale+0.25)
        #     self.entropyWeight+=0.0005
        # elif self.entropyWeight > 0.001:
        #     self.entropyWeight=max(0.001, self.entropyWeight-0.0002)
        # else:
        #     self.entropyWeight*=self.entropyDecay

        # # #Gradually decay by 0.25 as well!
        # if probabilityModifier and self.trgIntervalCounter >= 1:
        #     probabilityModifier.impactScale=min(1, probabilityModifier.impactScale+0.25)

        #Not sure why, but sometimes treats as 0.9999999 instead of 1 when verified outside of object
        if self.pretrainedScale > 0.999999:
            self.pretrainedScale = 1
        self.trgIntervalCounter+=1

        #Uncomment below for more granular logging
        #For calculating explained variance (how good critic is at predicting values)
        # flatReturns=np.concatenate(returnsArr)
        # flatCritic=np.concatenate(criticPredsArr)
        # explainedVariance = self.calcExplainedVariance(flatCritic,flatReturns)
        #wandbLogDict[f"{groupName}ExplainedVariance"] = explainedVariance
        #wandbLogDict[f"{groupName}AvgEntropy"] = sum(entropyArr) / len(entropyArr)
        #wandbLogDict[f"{groupName}AvgCriticVal"] = sum(criticValsArr) / len(criticValsArr)
        #wandbLogDict[f"{groupName}AvgCritLoss"] = sum(criticLossArr) / len(criticLossArr)
        #wandbLogDict[f"{groupName}AvgActLoss"] = sum(actorLossArr) / len(actorLossArr)
        #wandbLogDict[f"{groupName}AvgAdv"] = sum(advantage) / len(advantage)
        # #wandbLogDict[f"{groupName}NumTotalClips"] = numLowClips + numHighClips
        # #wandbLogDict[f"{groupName}NumLowClips"] = numLowClips
        # #wandbLogDict[f"{groupName}NumHighClips"] = numHighClips
        #print("Advantage is: ", advantage)
        # print(f"Average critic val: {sum(criticValsArr)/len(criticValsArr)} | Max critic val: {max(criticValsArr)} | Min critic val: {min(criticValsArr)}")
        # print(f"Average critic loss: {sum(criticLossArr)/len(criticLossArr)} | Max critic loss: {max(criticLossArr)} | Min critic loss: {min(criticLossArr)}")
        # print(f"Average actor loss: {sum(actorLossArr)/len(actorLossArr)} | Max actor loss: {max(actorLossArr)} | Min actor loss: {min(actorLossArr)}")
        # print(f"Average advantage: {sum(advantage)/len(advantage)} | Max advantage: {max(advantage)} | Min advantage: {min(advantage)}")

def genBatchIndexes(batchSize:int, dataSize:int):
    batch_start = np.arange(0, dataSize, batchSize)
    indices = np.arange(dataSize, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i:i+batchSize] for i in batch_start]
    return batches

#Class for reducing probabilities of actions
class ProbabilityModifier:
    def __init__(self, possibleActions, device):
        self.impactScale=0
        self.decayScale=0
        self.recommendedActions=[] #Keep track of hosts to act on (for applying same masking in learning as inference)
        self.possibleActions=possibleActions
        self.device=device
        self.hostCount={}
        self.hostsToNotMask=[]

    #Modify the probabilities in batches using previously collected hosts
    def modifyBatchProbabilities(self,dist,batch):
        actionMaskings=[[1 for i in range(0,len(self.possibleActions))] for j in range(0,len(batch))]
        for actionMaskingIndex in range(0,len(actionMaskings)):
            recommendedAction=self.recommendedActions[batch[actionMaskingIndex]]
            for actionIndex in range(0,len(self.possibleActions)):
                tmpAction=self.possibleActions[actionIndex]
                
                #Mask the host to act on if using action masking for LLM help type
                if recommendedAction and recommendedAction != actionIndex:
                    actionMaskings[actionMaskingIndex][actionIndex]=self.impactScale        
        modifiedProbs = dist.probs * torch.tensor(actionMaskings).to(self.device)
        modifiedProbs /= modifiedProbs.sum()#Renormalize probability distribution so sum=1
        return Categorical(modifiedProbs)

    #Reduce probabilities
    def modifyProbabilities(self,dist, recommendedAction=None):
        actionMasking=[1 for i in range(0,len(self.possibleActions))]
        self.recommendedActions.append(recommendedAction)
        for actionIndex in range(0,len(self.possibleActions)):
            if recommendedAction and recommendedAction != actionIndex:
                # pass #UNCOMMENT to only mask at training
                actionMasking[actionIndex]=self.impactScale

        #UNCOMMENT to keep track of hosts masked so far (if want to use that technique)
        # if hostToActOn in self.hostCount:
        #     self.hostCount[hostToActOn]+=1
        # else:
        #     self.hostCount[hostToActOn]=1
        #Can't directly multiply distribution with Categorical (need to get raw probabilities)
        maskedProbs = dist.probs * torch.tensor(actionMasking).to(self.device)
        maskedProbs /= maskedProbs.sum()
        #Now that have new masked probabilities, create new cateogorical distribution
        return Categorical(maskedProbs)

#Method to extract the number of hops for each host from YAML (for giving LLM additional context on scenario)
def extractHops(pathToScenario):
    with open(pathToScenario, "r") as file:
        scenarioDict=yaml.safe_load(file)
    graph=defaultdict(list)
    enterpriseHosts=set()
    for hostname in scenarioDict["Hosts"]:
        if "Enterprise" in hostname:
            enterpriseHosts.add(hostname)
        if "info" in scenarioDict["Hosts"][hostname]:
            for host in scenarioDict["Hosts"][hostname]["info"]:
                if host != hostname:
                    graph[host].append(hostname)
                    graph[hostname].append(host)
    for enterpriseHost in enterpriseHosts:
        for otherHost in enterpriseHosts:
            if otherHost != enterpriseHost:
                graph[enterpriseHost].append(otherHost)
                graph[otherHost].append(enterpriseHost)

    hops = {"Op_Server0": 0}
    queue = deque(["Op_Server0"])

    while queue:
        current = queue.popleft()
        current_hops = hops[current]

        for neighbor in graph[current]:
            if neighbor not in hops:  # Only visit unvisited nodes
                hops[neighbor] = current_hops + 1
                queue.append(neighbor)
    return hops

class PlayGame:
    def __init__(self,numEps:int=100,numSteps:int=32,batchSize:int=64,trgInterval=256,
               policyLR=1e-4,valueLR=1e-4,epochs=30,policyClip=0.2, policyGradClip=0.5, criticGradClip=0.1, 
               entropyCoef=0.01, entropyDecay=0.99, useTrueState=False,saveModelPath=False,
               saveToDB=False,hyperParameterTuning=False,llmHelp=None,llmPromptType="json"):
        self.numEps=numEps
        self.numSteps=numSteps
        self.batchSize=batchSize
        self.trgInterval=trgInterval
        self.policyLR=policyLR
        self.valueLR=valueLR
        self.epochs=epochs
        self.policyClip=policyClip
        self.entropyCoef=entropyCoef
        self.entropyDecay=entropyDecay
        self.policyGradClip=policyGradClip
        self.criticGradClip=criticGradClip
        self.useTrueState=useTrueState
        self.saveModelPath=saveModelPath
        self.hyperParameterTuning=hyperParameterTuning
        path=Path(__file__).parent.parent
        path=path/"CybORGModified/CybORG/Shared/Scenarios/modifiedScenarios/scenario_13hosts_3subnets.yaml"
        agents:dict={
            #'Red': B_lineAgent,
            'Red': B_lineAgent13,
            #'Red': RedMeanderAgent,
            'Green': GreenAgent,
        }
        self.recommendedAction=None
        self.recommendedDist=None
        #If don't specify agents in environment, they default to sleep agents
        self.cyborg:CybORG = CybORG(path,'sim',agents=agents)
        self.llmHelp=llmHelp
        #env:ChallengeWrapper = ChallengeWrapper(env=cyborg, agent_name='Blue')
        hostHops=extractHops(path)
        self.env:CustomBlueWrapper=CustomBlueWrapper(env=self.cyborg,useLLM=llmHelp, llmPromptType=llmPromptType,hostHops=hostHops)
        #If save to database is set then save to database!
        self.saveToDB=saveToDB
        if self.saveToDB:
            #Episode step data format: [redActions, blueActions, rewards, redStates, blueStates, stepNumber]
            self.episodeStepData=[]
            self.possibleRedActions=self.env.getActionMapping('Red')
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    def saveGraph(self,totalTrgTime,epRewards, numEps, numSteps, \
                  savePath=f'TuningGraph{int(time.time())}.png', \
                    title="PERFORMANCE OVER EPISODES"):
        plt.plot(epRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(title)
        description=f"""*Training took {totalTrgTime} seconds for {numEps} episodes and {numSteps} steps per episode.
        """
        plt.subplots_adjust(bottom=0.3)
        plt.text(-0.1, -0.3, description, transform=plt.gca().transAxes, fontsize=8, verticalalignment='center')
        plt.savefig(savePath)
        #plt.savefig(f'D:/Masters/Cyborg/Cage2/CageChallenge2LLM/ModifiedCybORG/figures/TuningGraph{int(time.time())}.png')
        plt.clf() #Clear figure

    def getBestActionAndDist(self,observation):
        dist,recommendedAction=self.env.getLLMRecommendation(returnAction=False, getDistribution=True)
        #Can't call sample since it's torch and not Categorical
        action=torch.multinomial(dist, num_samples=1)
        return action,dist.detach()#dist.probs.detach()
    
    def playRound(self, useTrueState=False, saveIteration=0, distilledSavePath="distilledLLMHost.pth"):
        #Uncomment for action masking
        # probabilityModifier=ProbabilityModifier(possibleActions=self.env.possibleActions, device=self.device)
        #Uncomment for aux loss only (not action masking too)
        probabilityModifier=None
        lastActions=[] #just for debugging
        startTime=time.time() #Calculate how long training takes
        actionSize=len(self.env.possibleActions) #Changing since self.env converts actionSpace into int (if want to do multiple runs)
        stateSize=self.env.observation_space.shape[0]
        if useTrueState: stateSize=len(self.env.getTrueState(vectorized=True))
        ppoAgent = Agent(actionSize=actionSize,stateSize=stateSize,policyLR=self.policyLR,criticLR=self.valueLR,
                        policyClip=self.policyClip,epochs=self.epochs, entropyCoef=self.entropyCoef, entropyDecay=self.entropyDecay, \
                            policyGradClip=self.policyGradClip, criticGradClip=self.criticGradClip)
        #Total steps exists outside loop since training can happen over multiple episodes
        totalSteps=0
        epRewardHistory=[]
        observations=[] #Get list of observations for LIME
        trainingData:list=[[],[],[],[],[],[]]
        recommendedDists=[]
        for ep in range(0,self.numEps):
            baseline=300
            currentReward=0
            if len(observations)>=300: break
            if self.saveToDB:
                #Episode step data format: [redActions, blueActions, rewards, redStates, blueStates, stepNumber]
                self.episodeStepData.append([[],[],[],[],[],[]])
            #Training data format: [state, action, oldProb, criticValue, reward, done]
            # trainingData:list=[[],[],[],[],[],[]]
            obs=self.env.reset()
            if useTrueState: obs=self.env.getTrueState(vectorized=True)
            epReward=0
            worstScore=0
            for step in range(0, self.numSteps):
                totalSteps+=1 #Calling at top so doesn't learn on first step
                dist,criticVal=ppoAgent.getAction(obs)

                # # # -----------FOR ACTION MASKING AND AUX LOSS (MAKE SURE AUX AND MASKING ARE DECREMENTED TOGETHER IN LEARN)
                # #Delete the probability modifier if impactscale is bigger or equal to 1
                # if (probabilityModifier and probabilityModifier.impactScale>=1):
                #     probabilityModifier=None
                #     self.recommendedAction=None

                # #Get the best host for masking
                # elif (probabilityModifier and probabilityModifier.impactScale<1):
                #     self.recommendedAction,self.recommendedDist=self.getBestActionAndDist(obs)
                #     # self.recommendedAction=self.env.getLLMRecommendation()
                #     recommendedDists.append(self.recommendedDist) 

                # #If probability modifier is None, ensure that hostToActOn is also none!
                # else:
                #     self.recommendedAction=None

                #-----------FOR JUST AUX LOSS
                if not probabilityModifier and ppoAgent.pretrainedScale < 1:
                    self.recommendedAction,self.recommendedDist=self.getBestActionAndDist(obs)
                    recommendedDists.append(self.recommendedDist) 

                maskedProbs=dist if not probabilityModifier else probabilityModifier.modifyProbabilities(dist,self.recommendedAction)
                action = maskedProbs.sample()

                #UNCOMENT TO HAVE NON-MASKED ORIG PROBS (for masking only at inference)
                prob=torch.squeeze(dist.log_prob(action)).item()

                action = torch.squeeze(action).item()
                criticVal = torch.squeeze(criticVal).item()
                nextObs,reward,done,_=self.env.step(action,timeStep=step)

                #Modifying reward in agent config instead of environment to improve modularity
                reward = (((reward + 13.1) / (13.1)) * 5) - 2.5
                if useTrueState: nextObs=self.env.getTrueState(vectorized=True)        
                #Add the output of the step to the training data
                bundledData=enumerate([obs,action,prob,criticVal,reward,done])
                for i, data in bundledData:
                    trainingData[i].append(data)

                #Learn from the training data
                if totalSteps % self.trgInterval == 0:
                    #Convert training data to np arrays (to support advanced indexing for batching)
                    for j in range(0,len(trainingData)):
                        trainingData[j]=np.array(trainingData[j])
                    #Add the batch indexes to the training data (last element)
                    trainingData.append(genBatchIndexes(self.batchSize,len(trainingData[0])))
                    ppoAgent.learn(trainingData, recommendedDists=recommendedDists, probabilityModifier=probabilityModifier)
                    recommendedDists=[]
                    #Clear the data
                    trainingData=[[],[],[],[],[],[]]
                obs=nextObs
                epReward+=reward
                if self.saveToDB:
                    #Just doing raw red action to stop overhead of preprocessing every step
                    self.episodeStepData[ep][0].append(str(self.env.get_last_action('Red')))
                    #self.episodeData[ep][0].append(self.possibleRedActions.index(str(self.env.get_last_action('Red'))))
                    #Vectorized Blue action taken
                    self.episodeStepData[ep][1].append(action)
                    #Float Step reward
                    self.episodeStepData[ep][2].append(reward)
                    #Vectorized True State
                    self.episodeStepData[ep][3].append(json.dumps(self.env.getTrueState(vectorized=True)))
                    #Vectorized Blue State (casting to list since nd array not serializable)
                    if useTrueState: 
                        self.episodeStepData[ep][4].append(json.dumps([0]*2*len(self.env.info)))
                    else:
                        self.episodeStepData[ep][4].append(json.dumps(obs.tolist()))
                    #Step number
                    self.episodeStepData[ep][5].append(step+1)
                if done: break
                #Just for debugging
                if ep > self.numEps-2: lastActions.append([str(self.env.env.environment_controller.get_last_action("Blue")),str(self.env.env.environment_controller.get_last_action("Red"))])


            #Save policy model for distillation!
            if ep >= 240:
                ppoAgent.savePolicyModel(f'LLMDistillations/StandardDist_0.5Temp.pth')

            epRewardHistory.append(epReward)
            #wandbLogDict[f"{groupName}EpReward"]=epReward
            print(f"Episode {ep} took {time.time()-startTime} seconds and got reward: {epReward} and total steps: {step+1}")
            #wandb.log(#wandbLogDict)
            #wandbLogDict.clear()
        if self.saveToDB:
            return self.episodeStepData,epRewardHistory
        if self.hyperParameterTuning:
            return epRewardHistory, time.time()-startTime

        print("Blue actions: ", [action[0] for action in lastActions])
        print("Red actions: ", [action[1] for action in lastActions])
        
        return epRewardHistory
if __name__=="__main__":
    batchSize=256
    trgInterval=256
    lr=0.0016
    criticLR=0.0016
    epochs=23
    policyClip=0.2
    entropyCoef=0.005
    entropyDecay=0.99
    criticGradClip=0.1

    newGame=PlayGame(numEps=504, llmHelp=True, llmPromptType=True, batchSize=batchSize,trgInterval=trgInterval,\
                     policyLR=lr,valueLR=criticLR,policyClip=policyClip, criticGradClip=criticGradClip, entropyCoef=entropyCoef)
    rewards=[]

    for y in range(0,1):
        groupName=f"changeToWandDB Group"
        for i in range(0,1):
            #wandb.init(project="ThesisFigures",group=groupName,name=f"run{i}",config={"Algorithm": groupName}, mode="offline", id=f"{groupName}_Run{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}")
            startTime=time.time()
            rewards.append(newGame.playRound(saveIteration=i, distilledSavePath=f'LLMDistillations/distilledLLMHost13.pth'))
            #wandb.finish()