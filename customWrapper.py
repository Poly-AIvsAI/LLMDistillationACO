import yaml
from gym import spaces
from pathlib import Path
import inspect
from copy import deepcopy
from prettytable import PrettyTable
import numpy as np
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Shared.Actions.AbstractActions import Impact, IsolateHost, UnIsolateHost
from collections import Counter
import sys, os
import math
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LLMIntegration.LLMAugmenter import LLMAugmenter

class CustomBlueWrapper(BaseWrapper):
    #Class variables for mapping the processed blue obs to corresponding activity and compromised values
    #SINCE MAPPINGS ARE SEQUENTIAL, LIST MAKES MORE SENSE
    # activityMapping=['None','Scan','Exploit']
    activityMapping={
        (0,0):"None",
        (1,0):"Scan",
        (1,1):"Exploit"
    }
    # compromisedMapping=['No','Unknown','User','Privileged']
    compromisedMapping={
        (0,0):"No",
        (0,1):"Unknown",
        (1,0):"User",
        (1,1):"Privileged"
    }
    # isolatedMapping=[False,True]
    isolatedMapping={
        0: "Unisolated",
        1: "Isolated"
    }

    patchedMapping={
        (0,0): "0",
        (0,1): "0-0.3",
        (1,0):"0.3-0.6",
        (1,1):"0.6-1"
    }

    trueStateKnownMapping=[False,True]
    trueStateScannedMapping=[False,True]
    trueStateAccessedMapping=['None','User','Privileged']
    def __init__(self,env,agentName='Blue', useLLM=None, llmPromptType="json", hostHops=None, origFeatureMapping=False):
        super().__init__(env,agentName)
        self.env=env
        self.agentName=agentName
        self.actionSignature={}
        self.baseline=None
        self.trueStateBaseline=None
        self.blueInfo={}
        self.info={}
        self.possibleActions=[] #List for mapping actions to number
        self.timeStep=0
        self.lastAction=0
        self.action_space=self.defineActionSpace()
        self.llmInfo={} #Dictionary for keeping track of what gets fed into LLM
        self.unformattedObs={} #Dictionary for keep track of hosts without formatting (for LLM)
        #If LLM is being used. Have to define before calling defineObservationSpace()
        self.useLLM=useLLM
        self.hostHops=hostHops if hostHops else None
        self.origFeatureMapping=origFeatureMapping #If using original feature mapping (not the updated one with floats, features for patch, isolate, etc)
        if useLLM:
            #FOR ACTION MASKING. Some LLMs seem to favor certain hosts based on names (e.g. enterprise1 > user1) so create generic mapping for the hosts (host1: user1, host2:op_server0, etc)
            self.genericHostMapping={f"host{i+1}": host for i, host in enumerate(env.environment_controller.state.hosts.keys())}
            #To map actions to generic names
            self.genericActionMapping={
                    "action1":"Analyse",
                    "action2":"Restore",
                    "action3":"Remove",
                    "action4":"Patch",
                    "action5":"IsolateHost",
                    "action6":"UnIsolateHost"
            }
            self.hostHops=None if not hostHops else {genericHost: hostHops[actualHost] for genericHost,actualHost in self.genericHostMapping.items() if actualHost in hostHops}
            #Convert back to original order
            if self.hostHops:
                self.hostHops=dict(sorted(self.hostHops.items(),key=lambda item: item[1]))
            possibleHosts=[host for host in self.genericHostMapping]
            self.possibleActionsStr=[str(possibleAction) for possibleAction in self.possibleActions] #For converting LLM response (string) to action index
            self.llmAugmenter=LLMAugmenter(self.possibleActionsStr,possibleHosts, llmPromptType)
            self.maxActionBits=math.ceil(math.log2(self.action_space.n))
            self.llmPromptType=llmPromptType #If LLM prompt is sentence or JSON based
        self.observation_space=self.defineObservationSpace()
        self.currentObs=[] #List for keeping track of the current observation of the agent
        self.scannedIps=set() #Set for keeping track of the scanned IPs by red agent
        self.impactOpCounter=0
        #Get the starting host PIDs for keeping track of what's a new process
        self.startingHostPIDs={host: [env.environment_controller.state.hosts[host].processes[i].pid \
                                 for i in range(0,len(env.environment_controller.state.hosts[host].processes))] \
                                    for host in env.environment_controller.state.hosts.keys()}
        self.newHostPIDs={host: [] for host in self.env.environment_controller.state.hosts.keys()}#Keep track of the PIDs (for remove)


    #Change stepsPerEp if changes in the future
    def step(self, action, timeStep=0, stepsPerEp=32):
        self.timeStep=self.timeStep+1 #Add 1 to timestep to make sure different than initial obs (0)
        processedAction=action
        action=self.possibleActions[action]#Change action to the actual action object
        #Cyborg's step returns result object - unpack before preprocessing obs
        result=self.env.step(agent=self.agentName,action=action)
        obs,reward,done,info=result.observation,result.reward,result.done,result.info
        if "Patch" in str(self.env.get_last_action('Blue')):
            reward=reward-0.3
        self.action_space=self.changeActionSpace(result.action_space)
        #Updating LLM info (passing in observation before preprocessing because want textual)
        self.updateLLMInfo(obs)

        obs=self.changeObservation(obs)
        self.currentObs=obs
        self.updateScannedIPs() #Update scanned IPs by red agent if applicable
        #This is just to add to agent for learning, so offsetting by 1 is okay (since will always be consistent from agent's view)
        self.lastAction=processedAction+1 #Add 1 to the last action to make sure it's different than no last action (0)        
        
        #Add done if end of episode (CHANGE IF END OF EPSIODE IS DIFFERENT)
        #Not messing with done since no terminating condition (just runs for x timesteps every episode)
        if self.timeStep>=stepsPerEp: 
            done=True
            self.timeStep=0

        #Changing it to decouple LLM's recommendation and step
        return obs,reward,done,info
       
    def getLLMRecommendation(self, returnAction=True, promptType="sentence", getDistribution=False):
        genericHostLLMInfo={genericHost: self.llmInfo[originalHost] for genericHost, originalHost in self.genericHostMapping.items() if originalHost != "timestep"}
        genericHostLLMInfo=self.convertLLMDictToText(genericHostLLMInfo) if promptType=="sentence" else genericHostLLMInfo
        #If returning single action from LLM
        if returnAction and not getDistribution:
            llmResponse=self.llmAugmenter.getBestAction(genericHostLLMInfo, hostHops=self.hostHops)
            bestHost=self.genericHostMapping[llmResponse[0]]
            bestAction=self.genericActionMapping[llmResponse[1]]
            llmResponse=f"{bestAction} {bestHost}"
            action=self.possibleActionsStr.index(llmResponse)
            return action
        #If returning distribution from LLM
        elif getDistribution:
            llmDist,llmResponse=self.llmAugmenter.getDistribution(genericHostLLMInfo, hostHops=self.hostHops)
            bestHost=self.genericHostMapping[llmResponse[0]]
            bestAction=self.genericActionMapping[llmResponse[1]]
            llmResponse=f"{bestAction} {bestHost}"
            action=self.possibleActionsStr.index(llmResponse)
            return llmDist,action
        else:
            llmResponse=self.llmAugmenter.getBestHost(genericHostLLMInfo, timestep=self.timeStep, hostHops=self.hostHops)
        return llmResponse

        
    #Method to convert the LLM info to text
    #For project, can either pass LLM info as dictionary (JSON) or as human structured text
    def convertLLMDictToText(self,llmInfo,specifyNone=True):
        #If specify the number of hops each host has, then rearrange order to match!
        if self.hostHops:
            hostsWithHops={host: llmInfo[host] for host in self.hostHops}
            hostsWithoutHops={host: llmInfo[host] for host in llmInfo if host not in self.hostHops and host!="timestep"}
            organizedHosts={**hostsWithHops, **hostsWithoutHops}
        else:
            organizedHosts={host: llmInfo[host] for host in llmInfo}
        textPrompt=""
        for host in organizedHosts:
            if host=="timestep": continue
            processes = ", ".join([f"{proc['count']} {'process' if proc['count']==1 else 'processes'} with: (Remote IP: {proc['remoteAddress']} and Port: {proc['localPort']})" for proc in organizedHosts[host]["Processes"]]) if "Processes" in organizedHosts[host] else False
            files= ", ".join([f"{file['Name']} at {file['Path']} (Density: {file['Density']}, Signed: {'Yes' if file['Signed'] else 'No'})" for file in organizedHosts[host]["Files"]]) if "Files" in organizedHosts[host] else False
            scans = ", ".join([f"{proc['count']} {'scan' if proc['count']==1 else 'scans'} with: (Remote IP: {proc['remoteAddress']} and Port: {proc['localPort']})" for proc in organizedHosts[host]["Scans"]]) if "Scans" in organizedHosts[host] else False
            if self.hostHops:
                # textPrompt+=(f"\n{host}| IP: {organizedHosts[host]['ip']}, "
                #             f"Isolated: {'Yes' if organizedHosts[host]['Isolated'] else 'No'}, Last Analysed: {'Never' if organizedHosts[host]['LastAnalysed']==-1 else organizedHosts[host]['LastAnalysed']}, ")
                 #If not only including host IP and isolated
                textPrompt+=(f"\n{host}| IP: {organizedHosts[host]['ip']}, "
                            f"{'ISOLATED' if organizedHosts[host]['Isolated'] else 'NOT ISOLATED'}")
            else:
                #If including priority and lastAnalysed
                textPrompt+=(f"\n{host}| IP: {organizedHosts[host]['ip']}, Priority: {organizedHosts[host]['Priority']}, "
                            f"{'ISOLATED' if organizedHosts[host]['Isolated'] else 'NOT ISOLATED'}, Last Analysed: {organizedHosts[host]['LastAnalysed']}")
                
            if files: 
                textPrompt+=f", Files: [{files}]"
            else: 
                textPrompt+=f", Files: []"

            if processes: 
                textPrompt+=f", Processes: [{processes}]"
            else:
                textPrompt+=f", Processes: []"
            
            if scans:
                textPrompt+=f", Scans: [{scans}]"
            else:
                textPrompt+=f", Scans: []"

        return textPrompt
    
    #Method to check if it's a scan or process (becausse PID isn't returned all the time)
    def checkIfProcessOrScanned(self,host,process):
        #If there are multiple instances of the same process, then return true
        if (process["Connections"][0]["local_port"],process["Connections"][0]["remote_address"])==self.discoveredProcess:
            return True
        #If PID not in starting PIDs, then it's a new process
        for currentProcess in self.env.environment_controller.state.hosts[host].processes:
            if currentProcess.pid not in self.startingHostPIDs[host] and currentProcess.pid not in self.newHostPIDs[host] \
                and process["Connections"][0]["local_port"]==currentProcess.get_state()[0].get("local_port"):
                self.newHostPIDs[host].append(currentProcess.pid)
                self.discoveredProcess=(process["Connections"][0]["local_port"],process["Connections"][0]["remote_address"])
                return True

    #Update the LLM information with the processes, files and sessions of each host.
    def updateLLMInfo(self,obs):
        # print("OBS IS: ", obs)
        #Make temporary observation to not overwrite it
        tmpObs=deepcopy(obs)
        #Update the unformatted observation too here
        for host in self.unformattedObs:
            if host in tmpObs:
                self.unformattedObs[host]=tmpObs[host]
            #Update isolated states for host
            self.unformattedObs[host]["Isolated"]=True if host in self.env.environment_controller.state.isolateList else False

        self.llmInfo["timestep"]=self.timeStep
        lastAction=str(self.env.get_last_action('Blue')) #For updating last analyzed
        for host in self.llmInfo:
            if host == "timestep": continue
            #Reset scan state (since not persistent across timesteps)
            if self.llmInfo[host]["Scans"]:
                self.llmInfo[host]["Scans"]=[]
            #If host was analyzed (LastAnalysed initialized to -1)
            if self.llmInfo[host]["LastAnalysed"]!=-1:
                #Set last analyzed to 0 if host was analyzed in last action else increase the count
                self.llmInfo[host]["LastAnalysed"]=0 if "Analyse" in lastAction and host in lastAction else self.llmInfo[host]["LastAnalysed"]+1
            elif "Analyse" in lastAction and host in lastAction:
                self.llmInfo[host]["LastAnalysed"]=0

            self.llmInfo[host]["Isolated"] = True if host in self.env.environment_controller.state.isolateList else False #Update the isolated status
            if host in tmpObs:
                for key in self.llmInfo[host]:
                    if key in tmpObs[host] and key != 'ip':
                        if key == 'Processes':
                            self.discoveredProcess=False
                            connectionCounter=Counter() #Counts if it's just a scan
                            processCounter=Counter() #Counts if process is actually spawned
                            for process in tmpObs[host][key]:
                                if isinstance(process,dict) and isinstance(process.get("Connections"),list):
                                    if self.checkIfProcessOrScanned(host,process):
                                        processInfo=(str(process["Connections"][0]["remote_address"]),process["Connections"][0]["local_port"])
                                        processCounter[processInfo]+=1
                                    #Only append connections if local address != remote address and no PID
                                    elif process["Connections"][0]["local_address"]!=process["Connections"][0]["remote_address"] and "PID" not in process:
                                        connectionInfo=(str(process["Connections"][0]["remote_address"]),process["Connections"][0]["local_port"])
                                        connectionCounter[connectionInfo]+=1
                                    elif "PID" in process:
                                        processInfo=(str(process["Connections"][0]["remote_address"]),process["Connections"][0]["local_port"])
                                        processCounter[processInfo]+=1
                            connInfo=[]
                            procInfo=[]
                            for conn, count in connectionCounter.items():
                                connInfo.append({"remoteAddress":conn[0],"localPort":conn[1],"count":count})
                            for proc, procCount in processCounter.items():
                                procInfo.append({"remoteAddress":proc[0],"localPort":proc[1],"count":procCount})

                            if connInfo: self.llmInfo[host]["Scans"]=connInfo
                            if procInfo: self.llmInfo[host]["Processes"]=procInfo
                        elif key=="Files":
                            files=[]
                            for file in tmpObs[host][key]:
                                files.append({"Name":file["File Name"],"Path":file["Path"],"Density":file["Density"],"Signed":file["Signed"]})
                            self.llmInfo[host][key]=files
                        else:
                            self.llmInfo[host][key] = tmpObs[host][key]

    def determineDone(self):
        #Since done is always set to false, set own terminating condition (Impact Op_Server0 3 times in a row)
        if str(self.env.get_last_action('Red'))=="Impact Op_Server0" and self.getTrueState(vectorized=False)["Op_Server0"]['Access']=='Privileged'\
            and str(self.env.get_last_action('Blue'))!="Restore Op_Server0" and str(self.env.get_last_action('Blue'))!="Remove Op_Server0":
            self.impactOpCounter+=1
            if self.impactOpCounter==3:
                self.impactOpCounter=0
                return True
        else:
            self.impactOpCounter=0
        return False
    
    def defineActionSpace(self):
        actionSpace=self.env.get_action_space(self.agentName)
        actionSpace=self.changeActionSpace(actionSpace)
        actionSpace = spaces.Discrete(actionSpace)
        return actionSpace

    def defineObservationSpace(self,agentName='Blue'):
        obs=self.reset(agentName)
        #If it's action masking then second argument will be host to mask:
        if self.useLLM=="actionMasking":
            obs=obs[0]
        return spaces.Box(-1.0,1.0,shape=(len(obs),),dtype=np.float32)

    def reset(self, agentName='Blue'):
        result=self.env.reset(agentName)
        obs = result.observation
        self.processInitialObs(obs)
        if self.useLLM=="actionMasking":
            obsTuple=self.changeObservation(obs,baseline=True,includeHostToMask=True)
            obs=obsTuple[0]
            hostToMask=obsTuple[1]
        else:
            obs = self.changeObservation(obs,baseline=True)
        self.currentObs=obs
        result.observation=obs
        #Get the true state as well for the baseline
        self.trueStateBaseline=self.env.get_agent_state('True')

        #Return the host to mask if actionMasking
        if self.useLLM=="actionMasking":
            return result.observation,hostToMask
        return result.observation

    #Method to add patch and isolate status to self.info
    def addPatchIsolateStatus(self):
        for host in self.info:
            self.info[host].append(1) if host in self.env.environment_controller.state.isolateList else self.info[host].append(0)
            self.info[host].append(self.env.environment_controller.state.patchedHosts[host])
        
    #Method copied from BlueTableWrapper to preprocess observation space to vector
    def changeObservation(self,observation,baseline=False,includeHostToMask=False):
        obs=deepcopy(observation)
        success = obs['success']
        self.processLastAction()
        #Only detect anomalies if not going against baseline (initial obs)
        anomalyObs = self.detectAnomalies(obs) if not baseline else obs
        del obs['success']
        info = self.processAnomalies(anomalyObs)
        #If it's the baseline set everything to default (not compromised, no activity)
        if baseline:
            for host in info:
                info[host][-2] = 'None'
                info[host][-1] = 'No'
                self.blueInfo[host][-1] = 'No'  
        self.info=info
        #Add the patch and isolate status for each host
        self.addPatchIsolateStatus()
        return self.preprocessObs(success,includeHostToMask=includeHostToMask)

    #Method copied from BlueTableWrapper to preprocess observation space to vector (_create_blue_table & _create_vector)
    def preprocessObs(self,success,includeHostToMask=False):
        table = PrettyTable([
            'Subnet',
            'IP Address',
            'Hostname',
            'Activity',
            'Compromised',
            'Isolated',
            'Patched'
            ])
        for hostid in self.info:
            table.add_row(self.info[hostid])
        
        table.sortby = 'Hostname'
        table.success = success
        protoVector=[]
        totalCompromised=0
        totalIsolated=0
        totalHosts=len(self.info)
        for row in table._rows:
            #Modified feature space to use floats and with integers (why have self.origFeatureMapping conditionals)
            # Activity
            activity = row[3]
            if activity == 'None':
                value=0.0 if not self.origFeatureMapping else [0,0]
            elif activity == 'Scan':
                value=0.3 if not self.origFeatureMapping else [1,0]
            elif activity == 'Exploit':
                value=1.0 if not self.origFeatureMapping else [1,1]
            else:
                raise ValueError('Table had invalid Access Level')
            
            if not self.origFeatureMapping:
                protoVector.append(value)
            else:
                protoVector.extend(value)
            # Compromised
            compromised = row[4]
            if compromised == 'No':
                value=0.0 if not self.origFeatureMapping else [0,0]
            elif compromised == 'Unknown':
                value=0.3 if not self.origFeatureMapping else [0,1]
            elif compromised == 'User':
                value=0.6 if not self.origFeatureMapping else [1,0]
                totalCompromised+=1
            elif compromised == 'Privileged':
                value=1.0 if not self.origFeatureMapping else [1,1]
                totalCompromised+=1
            else:
                raise ValueError('Table had invalid Access Level')
            if not self.origFeatureMapping:
                protoVector.append(value)
            else:
                protoVector.extend(value)
            if not self.origFeatureMapping:
                #Isolated
                isolated=row[5]
                if isolated:
                    totalIsolated+=1
                protoVector.append(float(isolated))
                #Patched
                patched=row[6]
                # #Append as float
                #Doing 1-patched so that 1 is worst (unpatched) and closer to 0 is best
                protoVector.append(patched)


        if not self.origFeatureMapping:
            #Add global features for number of hosts compromised (user/privileged) and number of hosts isolated
            protoVector.append(totalIsolated/totalHosts)
            protoVector.append(totalCompromised/totalHosts)
 
        return np.array(protoVector)
    
    #_detect_anomalies method from BlueTableWrapper
    def detectAnomalies(self,obs):
        if self.baseline is None:
            raise TypeError('BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method.')
        anomalyDict = {}
        #Return any files and processes that aren't included in the baseline
        for hostid,host in obs.items():
            if hostid == 'success': continue
            hostBaseline=self.baseline[hostid] #Get the baseline image of the host (to compare anomalies)
            if host==hostBaseline:continue #If nothing changed initially, no anomalies
            hostAnomalies={}
            if 'Files' in host:
                baselineFiles=hostBaseline.get('Files',[])
                anomalousFiles=[]
                for f in host['Files']:
                    if f not in baselineFiles:
                        anomalousFiles.append(f)
                if anomalousFiles:
                    hostAnomalies['Files']=anomalousFiles
            if 'Processes' in host:
                baselineProcesses=hostBaseline.get('Processes',[])
                anomalousProcesses=[]
                for p in host['Processes']:
                    if p not in baselineProcesses:
                        anomalousProcesses.append(p)
                if anomalousProcesses:
                    hostAnomalies['Processes']=anomalousProcesses
            if hostAnomalies:
                anomalyDict[hostid]=hostAnomalies
        return anomalyDict
    
    #_process_anomalies method from BlueTableWrapper
    def processAnomalies(self,anomalyDict):
        info = deepcopy(self.blueInfo)
        for hostid, hostAnomalies in anomalyDict.items():
            assert len(hostAnomalies) > 0, f'Anomaly dictionary for host {hostid} is empty - nothing to process'
            if 'Processes' in hostAnomalies:
                connectionType=self.interpretConnections(hostAnomalies['Processes'])
                info[hostid][-2]=connectionType
                if connectionType == "Exploit":
                    info[hostid][-1]='User'
                    self.blueInfo[hostid][-1]='User'
            if 'Files' in hostAnomalies:
                malware = [f['Density'] >= 0.9 for f in hostAnomalies['Files']]
                if any(malware):
                    info[hostid][-1] = 'Privileged'
                    self.blueInfo[hostid][-1] = 'Privileged'
        return info

    #_interpret_connections method from BlueTableWrapper
    def interpretConnections(self,activity):
        numConnections=len(activity)
        ports = set([item['Connections'][0]['local_port'] \
            for item in activity if 'Connections' in item])
        portFocus = len(ports)
        remotePorts = set([item['Connections'][0].get('remote_port') \
            for item in activity if 'Connections' in item])
        if None in remotePorts:
            remotePorts.remove(None)
        if numConnections >= 3 and portFocus >=3:
            anomaly = 'Scan'
        elif 4444 in remotePorts:
            anomaly = 'Exploit'
        elif numConnections >= 3 and portFocus == 1:
            anomaly = 'Exploit'
        elif 'Service Name' in activity[0]:
            anomaly = 'None'
        else:
            anomaly = 'Scan'
        return anomaly
    
    #Copied from _process_initial_obs from BlueTableWrapper
    def processInitialObs(self,obs):
        obs = obs.copy()
        self.baseline = obs
        del self.baseline['success']
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            interface = host['Interface'][0]
            subnet = interface['Subnet']
            ip = str(interface['IP Address'])
            hostname = host['System info']['Hostname']
            self.blueInfo[hostname] = [str(subnet),str(ip),hostname, 'None','No']
            #Initializing info fed into LLM. Can't set as sets since sets can't have dictionaries (non-hashable)
            self.llmInfo[hostname] = {"ip":ip,"Processes":[],"Files":[], "Scans": [], "Isolated": False, "LastAnalysed": -1}
            self.unformattedObs[hostname] = {"ip":ip,"Processes":"None","Files":"None", "Isolated": False}
            self.unformattedObs[hostname]["distribution"]=self.env.environment_controller.state.hosts[hostname].distribution
            self.unformattedObs[hostname]["kernel"]=self.env.environment_controller.state.hosts[hostname].kernel
            self.unformattedObs[hostname]["patches"]=self.env.environment_controller.state.hosts[hostname].patches
            self.unformattedObs[hostname]["hostname"]=self.env.environment_controller.state.hosts[hostname].hostname
            self.unformattedObs[hostname]["architecture"]=self.env.environment_controller.state.hosts[hostname].architecture
            self.unformattedObs[hostname]["respond_to_ping"]=self.env.environment_controller.state.hosts[hostname].respond_to_ping
            self.unformattedObs[hostname]["users"]=[str(user) for user in self.env.environment_controller.state.hosts[hostname].users]
            self.unformattedObs[hostname]["original_files"]=self.env.environment_controller.state.hosts[hostname].original_files
            self.unformattedObs[hostname]["original_sessions"]=self.env.environment_controller.state.hosts[hostname].original_sessions
            self.unformattedObs[hostname]["original_processes"]=[str(origProcess) for origProcess in self.env.environment_controller.state.hosts[hostname].original_processes]
            self.unformattedObs[hostname]["interfaces"]=[interface.name for interface in self.env.environment_controller.state.hosts[hostname].interfaces]
            #Set the priority of the host based on the Availability value in the scenario
            self.llmInfo[hostname]["Priority"]=\
                self.env.environment_controller.scenario._scenario['Hosts'][hostname]["AvailabilityValue"]\
                      if "AvailabilityValue" in self.env.environment_controller.scenario._scenario['Hosts'][hostname].keys()\
                          else "None"
                
        return self.blueInfo 

    #_process_last_action method from BlueTableWrapper
    def processLastAction(self):
        action=self.env.get_last_action('Blue')
        if action is not None:
            name=action.__class__.__name__
            hostname=action.get_params()['hostname'] if name in ('Restore', 'Remove') else None
            #Change host to not compromised if restored
            if name =='Restore':
                self.blueInfo[hostname][-1] = 'No'
                #Reset the LLM information too!
                self.llmInfo[hostname]["Processes"]=[]
                self.llmInfo[hostname]["Files"]=[]

                #Reset unfiltered ones:
                self.unformattedObs[hostname]["Processes"]="None"
                self.unformattedObs[hostname]["Files"]="None"
                
            #Change host to unknown if removed (and not compromised)
            elif name =='Remove':
                compromised = self.blueInfo[hostname][-1]
                if compromised != 'No':
                    self.blueInfo[hostname][-1] = 'Unknown'

                #Get Rid of process if its PID is no longer in the host's processes
                currentHostPIDs=[process.pid for process in self.env.environment_controller.state.hosts[hostname].processes]
                #If PID no longer in new PIDs, then remove it from self.llmInfo!
                for pid in self.newHostPIDs[hostname]:
                    if pid not in currentHostPIDs:
                        #Remove from LLM info and newHostPIDs dictionary
                        self.newHostPIDs[hostname].remove(pid)
                        self.llmInfo[hostname]["Processes"]=[]

    #Method copied from EnumActionWrapper to preprocess action space to vector
    def changeActionSpace(self, action_space: dict, agent="Blue") -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        possible_actions = []
        temp = {}
        params = ['action']

        #Go through each action and get the associated parameters
        for i, action in enumerate(action_space['action']):
            if action not in self.actionSignature:
                self.actionSignature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in self.actionSignature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)
                #For each parameter, get the possibilities and add them to the list
                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))
        #If returning red actions, return possible red actions instead of number of possible actions
        if agent=="Red": return possible_actions
        self.possibleActions = possible_actions
        return len(possible_actions)

    #This method processes true state using same bluetablewrapper method
    def getReadableProcessedState(self,trueState=False):
        obs={}
        if trueState:
            obs=self.env.environment_controller._filter_obs(self.env.environment_controller.get_true_state(self.env.environment_controller.INFO_DICT['True']))
            obs=self.changeObservation(obs.data)
        else:
            obs=self.currentObs
        hostCounter=0
        hosts=[host for host in self.info]
        hostinfo={}
        for i in range(0, len(obs), 7):
            activityEncoded=(obs[i],obs[i+1])
            mappedActivity=self.activityMapping[activityEncoded]
            
            compromisedEncoded=(obs[i+2],obs[i+3])
            mappedCompromised=self.compromisedMapping[compromisedEncoded]
            
            isolatedEncoded=obs[i+4]
            mappedIsolated=self.isolatedMapping[isolatedEncoded]

            patchedEncoded=(obs[i+5], obs[i+6])
            mappedPatched=self.patchedMapping[patchedEncoded]

            hostinfo[hosts[hostCounter]]={'Activity':mappedActivity,'Compromised':mappedCompromised,'Isolated':mappedIsolated,"Patched":mappedPatched}
            hostCounter+=1
        return hostinfo
    
    #This method processes the true state using the TrueTable method (using red agent's view as well)
    def getTrueState(self, vectorized=True):
        trueStateKnown={
            False: 0,
            True: 1
        }
        trueStateScanned={
            False: 0,
            True: 1
        }
        trueStateAccessed={
            'None': (0, 0),
            'User': (0, 1),
            'Privileged': (1, 1)
        }
        trueState=self.env.environment_controller._filter_obs(self.env.environment_controller.get_true_state(self.env.environment_controller.INFO_DICT['True'])).data
        success = trueState.pop('success')
        trueStateInfo=[] if vectorized else {}
        for hostid in trueState:
            host = trueState[hostid]
            for interface in host['Interface']:
                ip = interface['IP Address']
                if str(ip) == '127.0.0.1':
                    continue
                if 'Subnet' not in interface:
                    continue
                subnet = interface['Subnet']
                hostname = host['System info']['Hostname']
                action_space = self.env.get_action_space(agent = 'Red')
                known = action_space['ip_address'][ip]
                scanned = True if str(ip) in self.scannedIps else False
                access = self.determineRedAccess(host['Sessions'])
            if vectorized:
                trueStateInfo.extend(trueStateAccessed[access])
            else:
                trueStateInfo[hostname]={"Known":known,
                                        "Scanned":scanned,
                                        "Access":access}
        return trueStateInfo
    
    #Method copied from TrueTableWrapper to determine red's access
    def determineRedAccess(self,session_list):
        for session in session_list:
            if session['Agent'] != 'Red':
                continue
            privileged = session['Username'] in {'root','SYSTEM'}
            return 'Privileged' if privileged else 'User'

        return 'None'

    #Method copied from TrueTableWrapper to update scanned IPs by red
    def updateScannedIPs(self):
        action = self.get_last_action(agent='Red')
        if action.__class__.__name__ == 'DiscoverNetworkServices':
            red_obs = deepcopy(self.get_observation(agent='Red'))
            success = red_obs['success']
            if success:
                ip = red_obs.popitem()[0]
                self.scannedIps.add(ip)

    #Since action mappings are sequential just store them in list
    def getActionMapping(self,agent):
        actionSpace=self.get_action_space(agent)
        actionSpace=self.changeActionSpace(actionSpace,agent)
        actionMapping=[str(action) for action in actionSpace]
        return actionMapping
    def getStateMapping(self,agent):
        if agent=="True":
            return self.trueStateKnownMapping,self.trueStateScannedMapping,self.trueStateAccessedMapping
        if agent=="Blue":
            return self.activityMapping,self.compromisedMapping
    def getSubnets(self):
        subnets=set()
        for host in self.info: subnets.add(self.info[host][0])
        return list(subnets)
    def getHosts(self):
        hostNames=[hostName for hostName in self.info]
        ips=[self.info[host][1] for host in self.info]
        subnets=[self.info[host][0] for host in self.info]
        return hostNames,ips,subnets

    #Convert dictionary to easier to read table
    def createTable(self, dict):
        columnNames=[column for column in dict[list(dict.keys())[0]]]
        allNames=["Hostname"]
        allNames.extend(columnNames)
        table = PrettyTable(allNames)
        
        for host in dict:
            infoToAdd=[host]
            for column in columnNames:
                infoToAdd.append(dict[host][column])
            table.add_row(infoToAdd)
        
        # table.sortby = 'Hostname'
        return table