from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch  # Import torch to check for GPU
import re
from evaluate import load
from bert_score.utils import model2layers
import json
import random
from pathlib import Path

class LLMAugmenter:
        def __init__(self, possibleActions, possibleHosts, promptType, modelName="Vanessasml/cyber-risk-llama-3-8b", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        bertScorePath=Path(__file__).parent/"bertscore.py"):
                #modelName="segolilylabs/Lily-Cybersecurity-7B-v0.2" if promptType=="sentence" else "Vanessasml/cyber-risk-llama-3-8b"
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(modelName)
                self.model = AutoModelForCausalLM.from_pretrained(modelName).half().to(device)
                self.bertScore = load(bertScorePath)

                #To map actions to generic names
                self.genericActionMapping={
                        "Analyse": "action1",
                        "Restore": "action2",
                        "Remove": "action3",
                        "Patch": "action4",
                        "IsolateHost": "action5",
                        "UnIsolateHost": "action6"
                }
                
                # self.possibleActions=[f'{self.genericActionMapping[act.split(" ")[0]]} {act.split(" ")[1]}' for act in possibleActions]
                self.possibleActions=[f"action{i}" for i in range(1,int((len(possibleActions))/len(possibleHosts))+1)]
                # self.possibleActions=["Analyse", "Restore", "Remove", "Patch", "IsolateHost", "UnIsolateHost"]
                # self.possibleActions=possibleActions #COMMENT TO GO BACK TO GENERIC ACTION NAMES
                self.possibleHosts=possibleHosts
                self.promptType=promptType

        #Method to extract action or host. Putting both in single method since very similar process
        #i.e., find if LLM output is in list of possible actions/hosts, if not use bertscore to find closest match
        def extractActionOrHost(self,llmResponse,possibleOptions):
                possibleOptionFound=False
                ###The below extracts the first possibleOption found in the possibleOptions list (e.g., if the response is host4, host1 will return host1)
                # for possibleOption in possibleOptions:
                #         #Notice sometimes LLM adds extra spaces or messes with case, so using regex to get rid of all white space too
                #         processedOption=re.sub(r'\s+',' ',possibleOption.lower())
                #         processedResponse=re.sub(r'\s+',' ',llmResponse.lower())
                #         #Adding this escape regex to match entire word so that host1 doesn't match with host12 for example
                #         if re.search(rf'\b{re.escape(processedOption)}\b', processedResponse):
                #                 print("Recommend action: ",possibleOption)
                #                 possibleOptionFound=True
                #                 break
                
                ###The below extracts the first occurrence found the response (e.g., if the response is host4, host1 will return host4)
                processedResponse = re.sub(r'\s+', ' ', llmResponse.lower())
                matches = [(possibleOption, re.search(rf'\b{re.escape(possibleOption.lower())}\b', processedResponse).start())
                        for possibleOption in possibleOptions if re.search(rf'\b{re.escape(possibleOption.lower())}\b', processedResponse)]
                # print("MATCHES ARE: ", matches)
                if matches:
                        possibleOptionFound=True
                        # Find the option with the smallest start index (LLM seems to output host before)
                        possibleOption = min(matches, key=lambda x: x[1])[0]
                        #print("Recommend option: ", possibleOption)

                #If LLM output doesn't explicitly exist in list of possible actions, use bertscore to get the closest match!
                if not possibleOptionFound:
                        # print("NO POSSIBLE OPTION FOUND TRYING BERTSCORE!")
                        llmResponses=[llmResponse for i in range(0,len(possibleOptions))]
                        try:
                                results = self.bertScore.compute(
                                        predictions=llmResponses,
                                        references=possibleOptions,
                                        model_type="/home/users/tholl/.cache/huggingface/hub/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59",
                                        num_layers=model2layers["roberta-large"],
                                )
                        except Exception as e:
                                raise RuntimeError(f"Ensure you change the path of the preceding line to the roberta-large model in the compute function to the correct path where it is stored on your machine! Error details: {e}")

                        #Get max precision score from results (results is dictionary of lists)
                        #Enumerate to get the index along with the corresponding element (to extract from possibleActions)
                        optionIndex=max(enumerate(results["precision"]),key=lambda x: x[1])[0]
                        #print("Recommended option: ",self.possibleActions[optionIndex])
                        recommendedOption=possibleOptions[optionIndex]
                        return recommendedOption
                return possibleOption
        
        def generateLLMResponse(self,prompt):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # Move inputs to the same device (GPU if available)
                startTime = time.time()
                with torch.no_grad():
                        llmTokenIDs = self.model.generate(inputs.input_ids, 
                                                use_cache=True, #Caches to avoid redundant computations - think enabled by default 
                                                early_stopping=True, #Ensure stops at EOS token (instead of continuing)
                                                max_new_tokens=30, #Number of tokens generated after the input
                                                num_beams=1, #Number of generation paths followed 
                                                #temperature=0.5, #Introduce randomness in response - way longer and not increase in quality
                                                #do_sample=True, #Similar to temperate, introduces randomness - no noticed increased in quality
                                                num_return_sequences=1 #Generate single output sequence for single input
                                                )
                llmResponse = self.tokenizer.batch_decode(llmTokenIDs, #TokenIDs generated by LLM
                                                skip_special_tokens=True, #Remove special tokens like <eos>
                                                clean_up_tokenization_spaces=False #Retain spacing as produced by model
                                                )[0]
                
                #Only keep what was generated by LLM (strip original response)
                if len(llmResponse) > len(prompt):
                        #Start at len(prompt) and go until end of string
                        llmResponse=llmResponse[len(prompt):].strip()
                # print("Original LLM Response: ", llmResponse)
                return llmResponse

        #Method to return best action from LLM (for feature space modification)
        def getBestAction(self,prompt,hostHops=None,timestep=0):
                #Modify prompt differently depending on if JSON or sentence based
                if self.promptType=="json":
                        timestep=prompt.pop("timestep")
                        initialContext=f"You are defending a network at timestep {timestep}. Your goal is to keep the network fully functional. Choose an action from Restore, Patch, Isolate, Unisolate, Remove, Analyse. Only select actions that disrupt functionality if essential. Respond with <Action> <Hostname> only. Current network state: "
                        finalContext="Respond with exactly one action in the format: <Action> <Hostname>."
                        modifiedPrompt="\n"+initialContext+"\n"+json.dumps(prompt)+"\n"+finalContext+"\n"
                        prompt["timestep"]=timestep #Re-add the timestep after popping it
                        print("MODIFIED PROMPT: ", modifiedPrompt)
                        llmResponse=self.generateLLMResponse(modifiedPrompt)
                        print("RESPONSE 1: ", llmResponse)
                        bestAction=self.extractActionOrHost(llmResponse,self.possibleActions)
                        print("FINAL RESPONSE: ", bestAction)
                #If prompt is sentence
                elif self.promptType=="sentence":
                        initialContext=f"You are defending a network. Your goal is to keep the network fully functional. Choose an action from Restore, Patch, Isolate, Unisolate, Remove, Analyse. Only select actions that disrupt functionality if essential. Respond with <Action> <Hostname> only. Current network state: "
                        finalContext="Respond with exactly one action in the format: <Action> <Hostname>."
                        modifiedPrompt="\n"+initialContext+"\n"+prompt+"\n"+finalContext+"\n"
                        # print("MODIFIED PROMPT: ", modifiedPrompt)
                        llmResponse=self.generateLLMResponse(modifiedPrompt)
                        print("LLM RESPONSE: ", llmResponse)
                        bestAction=self.extractActionOrHost(llmResponse,self.possibleActions)
                        print("FINAL RESPONSE: ", bestAction)
                        bestHost=self.extractActionOrHost(llmResponse,self.possibleHosts)
                        bestAction=(bestHost,bestAction)
                else:
                        # return ("host1", "action1") #For debugging (IOT have it not go through LLM to increase debugging efficiency)
                        targetHost=next(iter(hostHops)) #The target is the first host in hosthops 
                        hopInformation=""
                        for host in hostHops:
                                hopInformation+=f"\n- {host}: {hostHops[host]} hops."

#----ANOTHER TRY ON THE PROMPT! - NEW STANDARD PROMPT!
                        initialContext=f'''
**Role**:
You are an expert at cybersecurity and network defense.

**Objective**:
You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

**Prioritize Hosts by Distance to the Target**
- Prioritize hosts that are closest to the operational server:{hopInformation}

**Determine the Best Action**
For the selected host, choose the best action based on the following definitions:
- **action1**: Use to gather additional information on a host, such as active processes, connections, or files.
- **action2**: Use to revert a fully compromised host where the attacker has gained persistence. 
- **action3**: Use to terminate non-persistent malicious processes before they escalate privileges or gain persistence.
- **action4**:Use to apply security updates and reduce the likelihood of future exploitation.
- **action5**: Use to block network access to and from a compromised host to prevent lateral movement.
- **action6**: Use to restore network access to a previously isolated host once it is deemed safe.

### **Current Network State**:'''

                        finalContext=f'''
Your response must follow this format exactly: <action> <hostname> (e.g., actionx hostx).

Best Action: 
'''


## MISTAKE OF A PROMPT (CURRENT STANDARD PROMPT) :(
#                         initialContext=f'''
# **Role**:
# You are an expert at cybersecurity and network defense.

# **Objective**:
# You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

# **Determine the Best Action**
# For the selected host, choose the best action based on the following definitions:
# - **action1**: Use to gather additional information on a host, such as active processes, connections, or files.
# - **action2**: Use to revert a fully compromised host where the attacker has gained persistence. 
# - **action3**: Use to terminate non-persistent malicious processes before they escalate privileges or gain persistence.
# - **action4**:Use to apply security updates and reduce the likelihood of future exploitation.
# - **action5**: Use to block network access to and from a compromised host to prevent lateral movement.
# - **action6**: Use to restore network access to a previously isolated host once it is deemed safe.

# ### **Current Network State**:'''

#                         finalContext=f'''
# Your response must follow this format exactly: <action> <hostname> (e.g., actionx hostx).

# Best Action: 
# '''
#---END CURRENT STANDARD OF PROMPT


#SECOND BEST PROMPT
#                         initialContext=f'''
# **Role**:
# You are an expert at cybersecurity and network defense.

# **Objective**:
# You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

# **Instructions**:
# Evaluate the Network State and follow these strict steps:

# ### **Step 1: Identify Critical Hosts**
# 🚨 **IMPORTANT:**  
# - Identify hosts with **suspicious processes or files**.

# ### **Step 2: Prioritize Hosts by Distance to the Target**
# - Prioritize hosts that are closest to the operational server:{hopInformation}

# ### **Step 3: Select the Best Host**
# - **{targetHost} can ONLY be selected if it has suspicious processes or files.**  
# - **IF AND ONLY IF** {targetHost} has **ANY suspicious processes or files**, select {targetHost}.  
# - **IF {targetHost} has NO suspicious processes or files, SKIP IT and proceed to Step 4.**

# ### **Step 4: Choose the closest host from the remaining list that has suspicious processes or files.** 

# ### **Step 5: Determine the Best Action**
# For the selected host, choose the best action based on the following rules:
# - **action1**: Use to gather additional information on a host, such as active processes, connections, or files.
# - **action2**: Use to revert a fully compromised host where the attacker has gained persistence. 
# - **action3**: Use to terminate non-persistent malicious processes before they escalate privileges or gain persistence.
# - **action4**:Use to apply security updates and reduce the likelihood of future exploitation.
# - **action5**: Use to block network access to and from a compromised host to prevent lateral movement.
# - **action6**: Use to restore network access to a previously isolated host once it is deemed safe.

# 🚨 **IP addresses are identifiers only. They must NOT be used for decision-making. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
# 🚨 **DO NOT output explanations or repeat instructions. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨

# ### **Current Network State**:'''

#                         finalContext=f'''
# Your response must follow this format exactly: <action> <hostname> (e.g., actionx hostx).

# Best Action: 
# '''


# # # # #-------------------UNREAL BEST PROMPT!!!!
#                         initialContext=f'''
# **Role**:
# You are an expert at cybersecurity and network defense.

# **Objective**:
# You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

# **Instructions**:
# Evaluate the Network State and follow these strict steps:

# ### **Step 1: Identify Critical Hosts**
# 🚨 **IMPORTANT:** 
# - **DO NOT select action2 or action3 unless there are suspicious processes or files.** 
# - Identify hosts with **suspicious processes or files**.

# ### **Step 2: Prioritize Hosts by Distance to the Target**
# - Prioritize hosts that are closest to the operational server:{hopInformation}

# ### **Step 3: Select the Best Host**
# - **{targetHost} can ONLY be selected if it has suspicious processes or files.**  
# - **IF AND ONLY IF** {targetHost} has **ANY suspicious processes or files**, select {targetHost}.  
# - **IF {targetHost} has NO suspicious processes or files, SKIP IT and proceed to Step 4.**

# ### **Step 4: Choose the closest host from the remaining list that has suspicious processes or files.** 

# ### **Step 5: Determine the Best Action**
# For the selected host, choose the best action based on the following rules:
# - **action1**: Use when a host's **status is unknown or unclear**, and additional information is needed to make a decision.
# - **action2**: Use **ONLY** if the host has suspicious files or long running processes. 
# - **action3**: Use **ONLY** if the host has short running processes.
# - **action4**: Use if the host has no current suspicious processes or files to prevent future exploitation.  
# - **action5**: Use if the host is NOT already isolated and has suspicious processes or files to prevent spreading to host8.
# - **action6**: Use if the host is already isolated AND has no suspicious processes or files.

# 🚨 **IP addresses are identifiers only. They must NOT be used for decision-making. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
# 🚨 **DO NOT output explanations or repeat instructions. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
# 🚨 **ONLY SELECT action2 or action3 if there are suspicious processes or files.** 🚨
# 🚨 **If the host has NO suspicious processes or files ("Files: []", "Processes: []"), action2 and action3 are NOT ALLOWED.** 🚨

# ### **Current Network State**:'''

# #SPECIFIC FOR 13 HOST ENV!
#                         finalContext=f'''
# Your response must follow this format exactly: <action> <hostname> (e.g., action4 host4).

# Best Action: 
# '''
# #                         finalContext=f'''
# # Your response must follow this format exactly: <action> <hostname> (e.g., actionx hostx).

# # Best Action: 
# # '''

# # # # #-------------------UNREAL BEST PROMPT!!!!                        
                        modifiedPrompt=initialContext+prompt+"\n"+finalContext

                        # print(modifiedPrompt)
                        # print(modifiedPrompt)
                        llmResponse=self.generateLLMResponse(modifiedPrompt)
                        # print("LLM RESPONSE: ", llmResponse)
                        
                        bestHost=self.extractActionOrHost(llmResponse,self.possibleHosts)
                        bestAction=self.extractActionOrHost(llmResponse,self.possibleActions)
                        
                        bestAction=(bestHost,bestAction)                        
                
                return bestAction
        
        #Method to return best host from LLM (for masking possible actions to best host)
        def getBestHost(self,prompt,timestep=0, additionalContext=None, hostHops=None):
                # return f"host{random.randint(1,10)}" #For debugging (IOT have it not go through LLM to increase debugging efficiency)
                if self.promptType=="json":
                        timestep=prompt.pop("timestep")
                        initialContext=f"You are defending a network at timestep {timestep}. Your goal is to keep the network fully functional. Choose which host requires the most attention."
                        finalContext="Respond with exactly one host to act on. Example: Op_Server1\nBest Host: "
                        modifiedPrompt="\n"+initialContext+"\n"+json.dumps(prompt)+"\n"+finalContext+"\n"
                        llmResponse=self.generateLLMResponse(modifiedPrompt)
                        bestHost=self.extractActionOrHost(llmResponse,self.possibleHosts)
                        prompt["timestep"]=timestep #Re-add the timestep after popping it
                        return bestHost
                #If there is additional context (# hops of each host to op_server, then add it!)
                elif hostHops:
                        targetHost=next(iter(hostHops))
                        hopInformation=""
                        for host in hostHops:
                                hopInformation+=f"\n- {host}: {hostHops[host]} hops."
                        #Two different prompts whether there are any processes or files
                        # if "Processes" in prompt or "Files" in prompt:
                        #         initialContext=f'\n**Role**:\nYou are an expert at cybersecurity and network defense. \n\n**Objective**:\nYou must choose **ONE host** to act on to maintain network functionality (e.g., hostx).\n\n**Instructions**:\nAnalyze the Network State and follow these strict steps:\n1. Step 1: Identify Which Hosts have Suspicious Processes or Files.\n\n2. Step 2: Prioritize Hosts by Distance. Among the suspicious hosts with suspicious processes or files, prioritize hosts that are closest to the target (i.e. the lowest hops):{hopInformation}\n\n3. Step 3: Analyze {targetHost}. \n- If {targetHost} has **ANY** suspicious processes or files, **SELECT** host8.\n- Else if {targetHost} does **NOT** have any suspicious processes or files, select a different host.\n\n**Current Network State**:'
                        #         finalContext=f"\n\nBest Host to Defend: \n"     
                                pass
                        else:
                                initialContext=f'\n**Role**:\nYou are an expert at cybersecurity and network defense. \n\n**Objective**:\nYou must choose **ONE host** to act on to maintain network functionality (e.g., hostx).\n\n**Special Rules**:\n1. If {targetHost} has **NO** suspicious processes or files do **NOT** select it under any circumstances.\n\n**Instructions**:\nAnalyze the Network State and follow these strict steps:\n1. Step 1: Identify Which Hosts have Suspicious Processes or Files.\n\n2. Step 2: Prioritize Hosts by Distance. Among the suspicious hosts with suspicious processes or files, prioritize hosts that are closest to the target (i.e. the lowest hops):{hopInformation}\n\n3. Step 3: Special Rule if No Suspicious Processes or Files: \n - Do **NOT** select {targetHost} unless {targetHost} has suspicious processes or files.  \n\n**Current Network State**:'
                                finalContext=f"\n\n**Special Rules**:\n1. If {targetHost} has **NO** suspicious processes or files do **NOT** select it under any circumstances.\n\nBest Host to Defend: \n"
                        modifiedPrompt=initialContext+prompt+finalContext
                        print("MODIFIED PROMPT: \n", modifiedPrompt)
                        llmResponse=self.generateLLMResponse(modifiedPrompt)
                        bestHost=self.extractActionOrHost(llmResponse,self.possibleHosts)
                        print("RECOMMENDED HOST TO ACT ON: ", bestHost)
                        return bestHost
                else:
                        #Modified initial and final context after asking chatGPT
                        #initialContext="You are an expert network defender. You must choose ONE host to defend to maintain network functionality.\nAnalyze the network state and respond with ONE host requiring immediate attention.\nCurrent Network State:"
                        #finalContext="Best Host: "
                        #initialContext=f"You are defending a network at timestep {timestep}. Your goal is to keep the network fully functional. Choose which host requires the most attention. Current network state:"
                        #finalContext="Respond with exactly ONE host to act on. Example: Host3.\nBest Host: "
                        #Found that ending with new line doesn't really help
                        #initialContext="You are an expert network defender. You must choose ONE host to defend to maintain network functionality.\nPrioritize hosts with benign files and processes.\nAnalyze the network state and respond with ONE host requiring immediate attention.\nCurrent Network State:"
                        #finalContext="Best Host To Defend: "
                        initialContext="You are an expert network defender. You must choose **ONE host** to defend to maintain network functionality.\n\n**Priority Rules**:\n1. Always prioritize hosts with suspicious files or processes, regardless of their network priority level.\n2. If no hosts have suspicious files or processes, defend the host with the highest network priority level.\n\nAnswer with a single host only (e.g., host5).\n\nAnalyze the network state and respond with **ONE host** requiring immediate attention.\n\nCurrent Network State:"
                        finalContext="\nBest Host To Defend: "
                        modifiedPrompt="\n"+initialContext+"\n"+prompt+"\n"+finalContext#+"\n"
                        print("MODIFIED PROMPT: \n", repr(modifiedPrompt))
                        llmResponse=self.generateLLMResponse(modifiedPrompt)
                        bestHost=self.extractActionOrHost(llmResponse,self.possibleHosts)
                        print("RECOMMENDED HOST TO ACT ON: ", bestHost)
                        return bestHost
        
        
        #-----METHOD TO GET DISTRIBUTION FROM LLM INSTEAD OF SINGLE ACTION :D
        def getDistribution(self,prompt,hostHops=None,timestep=0):
                # return ("host1", "action1") #For debugging (IOT have it not go through LLM to increase debugging efficiency)
                targetHost=next(iter(hostHops)) #The target is the first host in hosthops 
                hopInformation=""
                for host in hostHops:
                        hopInformation+=f"\n- {host}: {hostHops[host]} hops."


# # # #-------------------STANDARD PROMPT!!!!
                        initialContext=f'''
**Role**:
You are an expert at cybersecurity and network defense.

**Objective**:
You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

**Prioritize Hosts by Distance to the Target**
- Prioritize hosts that are closest to the operational server:{hopInformation}

**Determine the Best Action**
For the selected host, choose the best action based on the following definitions:
- **action1**: Use to gather additional information on a host, such as active processes, connections, or files.
- **action2**: Use to revert a fully compromised host where the attacker has gained persistence. 
- **action3**: Use to terminate non-persistent malicious processes before they escalate privileges or gain persistence.
- **action4**:Use to apply security updates and reduce the likelihood of future exploitation.
- **action5**: Use to block network access to and from a compromised host to prevent lateral movement.
- **action6**: Use to restore network access to a previously isolated host once it is deemed safe.

### **Current Network State**:'''

                        finalContext=f'''
Your response must follow this format exactly: <action> <hostname> (e.g., actionx hostx).

Best Action: 
'''
# # # #------------------- END STANDARD PROMPT!!!!
# # # # #-------------------UNREAL BEST PROMPT!!!!
#                         initialContext=f'''
# **Role**:
# You are an expert at cybersecurity and network defense.

# **Objective**:
# You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

# **Instructions**:
# Evaluate the Network State and follow these strict steps:

# ### **Step 1: Identify Critical Hosts**
# 🚨 **IMPORTANT:** 
# - **DO NOT select action2 or action3 unless there are suspicious processes or files.** 
# - Identify hosts with **suspicious processes or files**.

# ### **Step 2: Prioritize Hosts by Distance to the Target**
# - Prioritize hosts that are closest to the operational server:{hopInformation}

# ### **Step 3: Select the Best Host**
# - **{targetHost} can ONLY be selected if it has suspicious processes or files.**  
# - **IF AND ONLY IF** {targetHost} has **ANY suspicious processes or files**, select {targetHost}.  
# - **IF {targetHost} has NO suspicious processes or files, SKIP IT and proceed to Step 4.**

# ### **Step 4: Choose the closest host from the remaining list that has suspicious processes or files.** 

# ### **Step 5: Determine the Best Action**
# For the selected host, choose the best action based on the following rules:
# - **action1**: Use when a host's **status is unknown or unclear**, and additional information is needed to make a decision.
# - **action2**: Use **ONLY** if the host has suspicious files or long running processes. 
# - **action3**: Use **ONLY** if the host has short running processes.
# - **action4**: Use if the host has no current suspicious processes or files to prevent future exploitation.  
# - **action5**: Use if the host is NOT already isolated and has suspicious processes or files to prevent spreading to host8.
# - **action6**: Use if the host is already isolated AND has no suspicious processes or files.

# 🚨 **IP addresses are identifiers only. They must NOT be used for decision-making. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
# 🚨 **DO NOT output explanations or repeat instructions. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
# 🚨 **ONLY SELECT action2 or action3 if there are suspicious processes or files.** 🚨
# 🚨 **If the host has NO suspicious processes or files ("Files: []", "Processes: []"), action2 and action3 are NOT ALLOWED.** 🚨

# ### **Current Network State**:'''

# #SPECIFIC FOR 13 HOST ENV!
#                         finalContext=f'''
# Your response must follow this format exactly: <action> <hostname> (e.g., action4 host4).

# Best Action: 
# '''
#                         finalContext=f'''
# Your response must follow this format exactly: <action> <hostname> (e.g., actionx hostx).

# Best Action: 
# '''

# # # #-------------------END UNREAL BEST PROMPT!!!! 
                modifiedPrompt=initialContext+prompt+"\n"+finalContext
                inputs = self.tokenizer(modifiedPrompt, return_tensors="pt").to(self.device)  # Move inputs to the same device (GPU if available)
                #print("TOKEN COUNT: ", inputs.input_ids.shape[1])
                with torch.no_grad():
                        output = self.model.generate(inputs.input_ids, 
                                                use_cache=True, #Caches to avoid redundant computations - think enabled by default 
                                                early_stopping=True, #Ensure stops at EOS token (instead of continuing)
                                                max_new_tokens=25, #Number of tokens generated after the input
                                                num_beams=1, #Number of generation paths followed 
                                                #temperature=0.5, #Introduce randomness in response - way longer and not increase in quality
                                                #do_sample=True, #Similar to temperate, introduces randomness - no noticed increased in quality
                                                num_return_sequences=1, #Generate single output sequence for single input
                                                return_dict_in_generate=True, #Return generation output as a dictionary
                                                output_scores=True #Return scores for each token in the generated sequence
                                                )
                llmResponse = self.tokenizer.batch_decode(output.sequences, #TokenIDs generated by LLM
                                                skip_special_tokens=True, #Remove special tokens like <eos>
                                                clean_up_tokenization_spaces=False #Retain spacing as produced by model
                                                )[0]
                
                #Only keep what was generated by LLM (strip original response)
                if len(llmResponse) > len(modifiedPrompt):
                        #Start at len(prompt) and go until end of string
                        llmResponse=llmResponse[len(modifiedPrompt):].strip()
                
                #EXAMPLE OF GENERATED TOKENS FROM RESPONSE: ['action', '2', 'Ġhost', '4', 'Ċ', 'Explanation', ':', 'ĠĊ', 'The', 'Ġhost', '4', 'Ġhas', 'Ġsuspicious', 'Ġfiles', 'Ġand', 'Ġprocesses', ',', 'Ġso', 'Ġwe', 'Ġneed', 'Ġto', 'Ġtake', 'Ġaction', 'Ġto', 'Ġprevent']
                generatedIDs = output.sequences[0][inputs.input_ids.shape[1]:]
                generatedTokens = self.tokenizer.convert_ids_to_tokens(generatedIDs)
                #THIS IS RELYING ON FACT THAT ACTION AND HOST ARE ALWAYS FIRST 2 (REALLY 4) TOKENS
                hostIdxs=None
                actionIdxs=None
                for i, tok in enumerate(generatedTokens):
                        # Match action like ['action', '2'] or ['Ġaction', '3']
                        if tok.strip().lower() == 'action' and i + 1 < len(generatedTokens):
                                if generatedTokens[i + 1].isdigit():
                                        actionIdxs = (i, i + 1)

                # Match host like ['Ġhost', '4']
                if tok.strip().strip('Ġ').lower() == 'host' and i + 1 < len(generatedTokens):
                        if generatedTokens[i + 1].isdigit():
                                hostIdxs = (i, i + 1)

                actionIdxs=(0,1) if not actionIdxs else actionIdxs
                hostIdxs=(2,3) if not hostIdxs else hostIdxs
                
                #Get the logits for each of the first 4 tokens
                action_prefix_logits = output.scores[actionIdxs[0]][0]  #logits for first token, expecting "action"
                action_suffix_logits = output.scores[actionIdxs[1]][0]  #logits for action number: 1–7
                host_prefix_logits   = output.scores[hostIdxs[0]][0]  #logits for "Ġhost"
                host_suffix_logits   = output.scores[hostIdxs[1]][0]  #logits for host number: 1–13

                # actionTemp=2
                # hostTemp=4
                actionTemp=1
                hostTemp=1
                actionWordProbs = torch.nn.functional.softmax(action_prefix_logits/actionTemp, dim=-1) #Probs of action token (the word action)
                actionNumProbs = torch.nn.functional.softmax(action_suffix_logits/actionTemp, dim=-1) #Probs of action # (the number for the action)
                hostWordProbs   = torch.nn.functional.softmax(host_prefix_logits/hostTemp, dim=-1) #Probs of host token (the word host)
                hostNumProbs   = torch.nn.functional.softmax(host_suffix_logits/hostTemp, dim=-1) #Probs of host # (the number for the host)

                #Get token ID for 'action' and 'Ġhost'
                actionWordToken = self.tokenizer("action").input_ids[0]
                hostWordToken = self.tokenizer(" host").input_ids[0]  #Have to put space in front (ref ChatGPT for this one!)

                #Get token IDs for numbers 1–7 and 1–13 (action numbers, host numbers)
                actionNumTokens = {f"action{i}": self.tokenizer(str(i)).input_ids[0] for i in range(1, 7)}
                hostNumTokens = {f"host{i}": self.tokenizer(str(i)).input_ids[0] for i in range(1, 14)}
                
                #Final distributions over 7 actions and 13 hosts
                #This is done by multiplying the probability of action being the token by the corresponding action number (and same for hosts)
                actionDist = {
                label: actionWordProbs[actionWordToken].item() * actionNumProbs[token_id].item()
                for label, token_id in actionNumTokens.items()
                }

                hostDist = {
                label: hostWordProbs[hostWordToken].item() * hostNumProbs[token_id].item()
                for label, token_id in hostNumTokens.items()
        }

                # Normalize action_distribution
                totalActionProbs = sum(actionDist.values())
                for k in actionDist:
                        actionDist[k] /= totalActionProbs

                # Normalize host_distribution
                totalHostProbs = sum(hostDist.values())
                for k in hostDist:
                        hostDist[k] /= totalHostProbs


                #CREATE DISTRIBUTION FOR EVERY POSSIBLE ACTION TOO:
                jointDist = {}
                
                #Get all host and action names in dictionary with corresponding probabilities
                for action_name, p_action in actionDist.items():
                        for host_name, p_host in hostDist.items():
                                joint_key = f"{action_name} {host_name}"
                                jointDist[joint_key] = p_action * p_host
                # Normalize joint_distribution
                total=sum(jointDist.values())
                for k in jointDist:
                        jointDist[k] /= total
                # print("JOINT DISTRIBUTION: ", dict(sorted(joint_distribution.items(), key=lambda item: item[1], reverse=True)))
                # print("JOINT DISTRIBUTION: ",joint_distribution)

                #Change the order of the dictionary so that action3 comes after action1 (changing the order in which definitions were presented showed small increase in performance)
                #REF CHATGPT FOR THE QUICK CODE TO DO THIS DICT REORDERING :D
                custom_action_order = ['action1', 'action3', 'action2', 'action4', 'action5', 'action6', 'action7']
                action_rank = {action: i for i, action in enumerate(custom_action_order)}
                sortedJointDist = dict(sorted(jointDist.items(),key=lambda x: (action_rank[x[0].split()[0]], int(x[0].split()[1].replace('host', '')))))

                #Convert from dictionary into torch that can actually be used in RL :D
                finalDist=torch.zeros(len(sortedJointDist))
                for i, (k,v) in enumerate(sortedJointDist.items()):
                        finalDist[i]=v
               

                #If want to sharpen dist (lower temperature score = sharper, higher than 1 = flatter)
                temperature=0.5
                logProbs=torch.log(finalDist.clamp(min=1e-8)) #Avoid log(0)
                sharpenedDist=torch.softmax(logProbs/temperature, dim=-1)

                bestHost=self.extractActionOrHost(llmResponse,self.possibleHosts)
                bestAction=self.extractActionOrHost(llmResponse,self.possibleActions)
                
                bestAction=(bestHost,bestAction)  
                return sharpenedDist, bestAction


















#For testing out custom prompts
prompt="""
You are defending a network at timestep 20. Your goal is to keep the network fully functional. Choose which host that requires the most attention. Current network state:
Host1 | IP: 10.0.243.186, Isolated: No, Last Analysed: 21 steps ago
Host2 | IP: 10.0.75.228, Isolated: No, Last Analysed: 15 steps ago, Files: [stuxnet.exe at / (Density: 0.8, Signed: No), auth.log at /var/logs (Density: 0.2, Signed: Yes)], Processes: [4 processes with: (Remote IP: 10.0.221.177 and Port: 443), 4 processes with: (Remote IP: 10.0.221.187 and Port: 22)]
Host3 | IP: 10.0.98.74, Isolated: No, Last Analysed: 13 steps ago, Files: [auth.log at /var/logs (Density: 0.2, Signed: Yes), auth.log at /var/logs (Density: 0.2, Signed: Yes)]
Host4 | IP: 10.0.75.232, Isolated: No, Last Analysed: 4 steps ago, Files: [escalate.sh at /tmp/ (Density: 0.9, Signed: No)], Processes: [3 processes with: (Remote IP: 10.0.221.182 and Port: 22)]
Host5 | IP: 10.0.243.181, Isolated: No, Last Analysed: 23 steps ago
Host6 | IP: 10.0.98.76, Isolated: No, Last Analysed: 22 steps ago
Host7 | IP: 10.0.243.183, Isolated: No, Last Analysed: 19 steps ago
Host8 | IP: 10.0.75.226, Isolated: No, Last Analysed: Never
Host9 | IP: 10.0.98.71, Isolated: No, Last Analysed: 21 steps ago, Files: [stuxnet.exe at / (Density: 0.8, Signed: No)], Processes: [2 processes with: (Remote IP: 10.0.221.179 and Port: 21), 3 processes with: (Remote IP: 10.0.221.180 and Port: 3389), 2 processes with: (Remote IP: 10.0.221.179 and Port: 22)]
Host10 | IP: 10.0.75.225, Isolated: No, Last Analysed: 20 steps ago
Respond with exactly ONE host to act on. Example: Host3.
Best Host:
"""

prompt2="""
You are defending a network at timestep 20. Your goal is to keep the network fully functional. Choose an action from Restore, Patch, Isolate, Unisolate, Remove, Analyse. Only select actions that disrupt functionality if essential. Respond with <Action> <Hostname> only. Current network state: 
{'User1': {'ip': '10.0.240.148', 'Processes': [{'remoteAddress': '10.0.240.154', 'localPort': 22, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 21, 'count': 1}], 'Files': [], 'Isolated': False, 'LastAnalysed': 0, 'Priority': 'None'}, 'User2': {'ip': '10.0.240.146', 'Processes': [{'remoteAddress': '10.0.240.154', 'localPort': 445, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 139, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 135, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 3389, 'count': 1}], 'Files': [{'Name': 'cmd.exe', 'Path': 'C:\\temp\\', 'Density': 0.9, 'Signed': False}], 'Isolated': False, 'LastAnalysed': 3, 'Priority': 'None'}, 'User3': {'ip': '10.0.240.152', 'Processes': [{'remoteAddress': '10.0.240.154', 'localPort': 80, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 3389, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 443, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 25, 'count': 1}], 'Files': [], 'Isolated': False, 'LastAnalysed': 31, 'Priority': 'None'}, 'User4': {'ip': '10.0.240.149', 'Processes': [{'remoteAddress': '10.0.240.154', 'localPort': 22, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 80, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 3390, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 443, 'count': 1}, {'remoteAddress': '10.0.240.154', 'localPort': 25, 'count': 1}], 'Files': [], 'Isolated': False, 'LastAnalysed': 4, 'Priority': 'None'}
Respond with exactly one action in the format: <Action> <Hostname>.
"""

prompt='''
**Role**:
You are an expert network defender. 

**Objective**:
You must choose **ONE host** to act on to maintain network functionality (e.g., hostx).

**Key Rule**:  
If host8 has any suspicious processes or files, it is the highest priority, regardless of other hosts' state or proximity in the network. In this case, you must act on host8 directly.

**Instructions**:
Analyze the network state and follow these strict steps:
Step 1: Analyze host8. Check if host8 has any suspicious processes or files?
- If host8 has any suspicious processes or files, act on host8 immediately.
- If host8 does not have any suspicious processes or files, proceed to step 2.
Step 2: Analyze Threat Propagation. Identify which hosts can lead the attacker to host8.
Step 3: Analyze Suspicious Hosts. Among the hosts identified in step 3, which have suspicious processes or files?
- If multiple have suspicious processes or files, prioritize hosts that are closer to host8 in the network structure.
- If none have suspicious processes or files, prioritize hosts that are farther from host8 in the network structure.
Step 4: Make a Final Decision. Select the SINGLE most critical host to act on based on steps 1 to 4 (e.g., host2).

**Network Propagation Rules**:
1. The attacker starts from hosts in subnetA (hosts 9, 10, 11, 12, 13) connected to the internet
2. From subnet A the attacker can propagate to subnet B (hosts 1, 2, 3, 4)
3. From subnet B the attacker can propagate to subnet C (hosts 5, 6, 7, 8)
4. Host 8 is the attacker's target.
*Connections*: 
- Host 4 -> Host 8
- Hosts 12, 13 -> Host 2
- Hosts 10, 11 -> Host 3
*Example of attack path*:
- host10 -> host3 -> host4 -> host8

**Current Network State**:
host1| IP: 10.0.138.76, Isolated: No, Last Analysed: -1
host2| IP: 10.0.138.73, Isolated: No, Last Analysed: 3
host3| IP: 10.0.138.65, Isolated: No, Last Analysed: -1, Files: [cmd.exe at C:\\temp\\ (Density: 0.9, Signed: No), escalate.exe at C:\\temp\\ (Density: 0.9, Signed: No)], Processes: [1 process with: (Remote IP: 10.0.174.13 and Port: 22), 1 process with: (Remote IP: 10.0.174.13 and Port: 135), 1 process with: (Remote IP: 10.0.174.13 and Port: 3389), 1 process with: (Remote IP: 10.0.174.13 and Port: 445), 1 process with: (Remote IP: 10.0.174.13 and Port: 139), 1 process with: (Remote IP: 10.0.174.13 and Port: 80), 1 process with: (Remote IP: 10.0.174.13 and Port: 443)]
host4| IP: 10.0.138.66, Isolated: No, Last Analysed: 5, Processes: [1 process with: (Remote IP: 10.0.174.13 and Port: 443), 1 process with: (Remote IP: 10.0.174.13 and Port: 53385)]
host5| IP: 10.0.112.71, Isolated: No, Last Analysed: -1
host6| IP: 10.0.112.65, Isolated: No, Last Analysed: -1
host7| IP: 10.0.112.78, Isolated: No, Last Analysed: -1
host8| IP: 10.0.112.74, Isolated: No, Last Analysed: -1
host9| IP: 10.0.174.13, Isolated: No, Last Analysed: -1
host10| IP: 10.0.174.6, Isolated: No, Last Analysed: -1
host11| IP: 10.0.174.7, Isolated: No, Last Analysed: -1, Processes: [1 process with: (Remote IP: 10.0.174.13 and Port: 445), 1 process with: (Remote IP: 10.0.174.13 and Port: 139), 1 process with: (Remote IP: 10.0.174.13 and Port: 135), 1 process with: (Remote IP: 10.0.174.13 and Port: 3389)]
host12| IP: 10.0.174.10, Isolated: No, Last Analysed: -1, Processes: [1 process with: (Remote IP: 10.0.174.13 and Port: 80), 1 process with: (Remote IP: 10.0.174.13 and Port: 3389), 1 process with: (Remote IP: 10.0.174.13 and Port: 443), 1 process with: (Remote IP: 10.0.174.13 and Port: 25)]
host13| IP: 10.0.174.11, Isolated: No, Last Analysed: -1, Processes: [1 process with: (Remote IP: 10.0.174.13 and Port: 22), 1 process with: (Remote IP: 10.0.174.13 and Port: 80), 1 process with: (Remote IP: 10.0.174.13 and Port: 3390), 1 process with: (Remote IP: 10.0.174.13 and Port: 443), 1 process with: (Remote IP: 10.0.174.13 and Port: 25)]

Best Host To Defend (e.g., restore hostx): 
'''

#Method to return best action from LLM (for feature space modification)
def testLLMResponse(prompt,model,tokenizer,device):
        prompt='''
**Role**:
You are an expert at cybersecurity and network defense.

**Objective**:
You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

**Instructions**:
Evaluate the Network State and follow these strict steps:

### **Step 1: Identify Critical Hosts**
🚨 **IMPORTANT:** 
- **DO NOT select action2 or action3 unless there are suspicious processes or files.** 
- Identify hosts with **suspicious processes or files**.

### **Step 2: Prioritize Hosts by Distance to the Target**
- Prioritize hosts that are closest to the operational server:
- host8: 0 hops.
- host4: 1 hops.
- host2: 2 hops.
- host3: 2 hops.
- host10: 3 hops.
- host11: 3 hops.
- host12: 3 hops.
- host13: 3 hops.

### **Step 3: Select the Best Host**
- **host8 can ONLY be selected if it has suspicious processes or files.**  
- **IF AND ONLY IF** host8 has **ANY suspicious processes or files**, select host8.  
- **IF host8 has NO suspicious processes or files, SKIP IT and proceed to Step 4.**

### **Step 4: Choose the closest host from the remaining list that has suspicious processes or files.** 

### **Step 5: Determine the Best Action**
For the selected host, choose the best action based on the following rules:
- **action1**: Use when a host's **status is unknown or unclear**, and additional information is needed to make a decision.
- **action2**: Use **ONLY** if the host has suspicious files or long running processes. 
- **action3**: Use **ONLY** if the host has short running processes.
- **action4**: Use if the host has no current suspicious processes or files to prevent future exploitation.  
- **action5**: Use if the host is NOT already isolated and has suspicious processes or files to prevent spreading to host8.
- **action6**: Use if the host is already isolated AND has no suspicious processes or files.

🚨 **IP addresses are identifiers only. They must NOT be used for decision-making. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
🚨 **DO NOT output explanations or repeat instructions. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
🚨 **ONLY SELECT action2 or action3 if there are suspicious processes or files.** 🚨
🚨 **If the host has NO suspicious processes or files ("Files: []", "Processes: []"), action2 and action3 are NOT ALLOWED.** 🚨

### **Current Network State**:
host8| IP: 10.0.108.129, NOT ISOLATED, Files: [], Processes: [11 processes with: (Remote IP: 10.0.146.86 and Port: 22)], Scans: []
host4| IP: 10.0.16.133, NOT ISOLATED, Files: [cmd.exe at C:\temp\ (Density: 0.9, Signed: No), escalate.exe at C:\temp\ (Density: 0.9, Signed: No)], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 55497)], Scans: []
host2| IP: 10.0.16.134, NOT ISOLATED, Files: [], Processes: [], Scans: []
host3| IP: 10.0.16.131, NOT ISOLATED, Files: [], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 51248)], Scans: []
host10| IP: 10.0.146.93, NOT ISOLATED, Files: [], Processes: [], Scans: []
host11| IP: 10.0.146.90, NOT ISOLATED, Files: [cmd.exe at C:\temp\ (Density: 0.9, Signed: No), escalate.exe at C:\temp\ (Density: 0.9, Signed: No)], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 56133)], Scans: []
host12| IP: 10.0.146.85, NOT ISOLATED, Files: [], Processes: [], Scans: [1 scan with: (Remote IP: 10.0.146.86 and Port: 80), 1 scan with: (Remote IP: 10.0.146.86 and Port: 3389), 1 scan with: (Remote IP: 10.0.146.86 and Port: 443), 1 scan with: (Remote IP: 10.0.146.86 and Port: 25)]
host13| IP: 10.0.146.87, NOT ISOLATED, Files: [], Processes: [], Scans: []
host1| IP: 10.0.16.138, NOT ISOLATED, Files: [], Processes: [], Scans: []
host5| IP: 10.0.108.131, NOT ISOLATED, Files: [], Processes: [], Scans: []
host6| IP: 10.0.108.134, NOT ISOLATED, Files: [], Processes: [], Scans: []
host7| IP: 10.0.108.137, NOT ISOLATED, Files: [], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 22)], Scans: []
host9| IP: 10.0.146.86, NOT ISOLATED, Files: [], Processes: [], Scans: []

Your response must follow this format exactly: <action> <hostname> (e.g., action4 host4).

Best Action: 
'''
        #timestep=prompt.pop("timestep")
        #initialContext=f"You are defending a network at timestep {timestep}. Your goal is to keep the network fully functional. Choose an action from Restore, Patch, Isolate, Unisolate, Remove, Analyse. Only select actions that disrupt functionality if essential. Respond with <Action> <Hostname> only. Current network state: "
        finalContext="Respond with exactly one action in the format: <Action> <Hostname>."
        #modifiedPrompt="\n"+initialContext+"\n"+json.dumps(prompt)+"\n"+finalContext+"\n"
        modifiedPrompt=prompt
        print("Modified prompt: ", repr(modifiedPrompt))
        inputs = tokenizer(modifiedPrompt, return_tensors="pt").to(device)  # Move inputs to the same device (GPU if available)
        #print("TOKEN COUNT: ", inputs.input_ids.shape[1])
        startTime = time.time()
        with torch.no_grad():
                llmTokenIDs = model.generate(inputs.input_ids, 
                                        use_cache=True, #Caches to avoid redundant computations - think enabled by default 
                                        early_stopping=True, #Ensure stops at EOS token (instead of continuing)
                                        max_new_tokens=25, #Number of tokens generated after the input
                                        num_beams=1, #Number of generation paths followed 
                                        #temperature=0.5, #Introduce randomness in response - way longer and not increase in quality
                                        #do_sample=True, #Similar to temperate, introduces randomness - no noticed increased in quality
                                        num_return_sequences=1 #Generate single output sequence for single input
                                        )
        llmResponse = tokenizer.batch_decode(llmTokenIDs, #TokenIDs generated by LLM
                                        skip_special_tokens=True, #Remove special tokens like <eos>
                                        clean_up_tokenization_spaces=False #Retain spacing as produced by model
                                        )[0]
        
        #Only keep what was generated by LLM (strip original response)
        if len(llmResponse) > len(modifiedPrompt):
                #Start at len(prompt) and go until end of string
                llmResponse=llmResponse[len(modifiedPrompt):].strip()
        print("LLM Response: ", llmResponse)
        #prompt["timestep"]=timestep #Re-add the timestep after popping it



#METHOD TO TRY AND EXTRACT PROBABILITIES FOR EVERY POSSIBLE ACTION/HOST
#Method to return best action from LLM (for feature space modification)
def extractProbabilities(llm,prompt):
        tokenizer=llm.tokenizer
        model=llm.model
        device=llm.device
        possibleActions=llm.possibleActions
        possibleHosts=llm.possibleHosts

        prompt='''
**Role**:
You are an expert at cybersecurity and network defense.

**Objective**:
You must choose **ONE ACTION** to execute on the most critical host to maintain network functionality.

**Instructions**:
Evaluate the Network State and follow these strict steps:

### **Step 1: Identify Critical Hosts**
🚨 **IMPORTANT:** 
- **DO NOT select action2 or action3 unless there are suspicious processes or files.** 
- Identify hosts with **suspicious processes or files**.

### **Step 2: Prioritize Hosts by Distance to the Target**
- Prioritize hosts that are closest to the operational server:
- host8: 0 hops.
- host4: 1 hops.
- host2: 2 hops.
- host3: 2 hops.
- host10: 3 hops.
- host11: 3 hops.
- host12: 3 hops.
- host13: 3 hops.

### **Step 3: Select the Best Host**
- **host8 can ONLY be selected if it has suspicious processes or files.**  
- **IF AND ONLY IF** host8 has **ANY suspicious processes or files**, select host8.  
- **IF host8 has NO suspicious processes or files, SKIP IT and proceed to Step 4.**

### **Step 4: Choose the closest host from the remaining list that has suspicious processes or files.** 

### **Step 5: Determine the Best Action**
For the selected host, choose the best action based on the following rules:
- **action1**: Use when a host's **status is unknown or unclear**, and additional information is needed to make a decision.
- **action2**: Use **ONLY** if the host has suspicious files or long running processes. 
- **action3**: Use **ONLY** if the host has short running processes.
- **action4**: Use if the host has no current suspicious processes or files to prevent future exploitation.  
- **action5**: Use if the host is NOT already isolated and has suspicious processes or files to prevent spreading to host8.
- **action6**: Use if the host is already isolated AND has no suspicious processes or files.

🚨 **IP addresses are identifiers only. They must NOT be used for decision-making. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
🚨 **DO NOT output explanations or repeat instructions. The correct action is determined ONLY by the host's state and proximity (i.e., number of hops).** 🚨
🚨 **ONLY SELECT action2 or action3 if there are suspicious processes or files.** 🚨
🚨 **If the host has NO suspicious processes or files ("Files: []", "Processes: []"), action2 and action3 are NOT ALLOWED.** 🚨

### **Current Network State**:
host8| IP: 10.0.108.129, NOT ISOLATED, Files: [], Processes: [11 processes with: (Remote IP: 10.0.146.86 and Port: 22)], Scans: []
host4| IP: 10.0.16.133, NOT ISOLATED, Files: [cmd.exe at C:\temp\ (Density: 0.9, Signed: No), escalate.exe at C:\temp\ (Density: 0.9, Signed: No)], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 55497)], Scans: []
host2| IP: 10.0.16.134, NOT ISOLATED, Files: [], Processes: [], Scans: []
host3| IP: 10.0.16.131, NOT ISOLATED, Files: [], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 51248)], Scans: []
host10| IP: 10.0.146.93, NOT ISOLATED, Files: [], Processes: [], Scans: []
host11| IP: 10.0.146.90, NOT ISOLATED, Files: [cmd.exe at C:\temp\ (Density: 0.9, Signed: No), escalate.exe at C:\temp\ (Density: 0.9, Signed: No)], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 56133)], Scans: []
host12| IP: 10.0.146.85, NOT ISOLATED, Files: [], Processes: [], Scans: [1 scan with: (Remote IP: 10.0.146.86 and Port: 80), 1 scan with: (Remote IP: 10.0.146.86 and Port: 3389), 1 scan with: (Remote IP: 10.0.146.86 and Port: 443), 1 scan with: (Remote IP: 10.0.146.86 and Port: 25)]
host13| IP: 10.0.146.87, NOT ISOLATED, Files: [], Processes: [], Scans: []
host1| IP: 10.0.16.138, NOT ISOLATED, Files: [], Processes: [], Scans: []
host5| IP: 10.0.108.131, NOT ISOLATED, Files: [], Processes: [], Scans: []
host6| IP: 10.0.108.134, NOT ISOLATED, Files: [], Processes: [], Scans: []
host7| IP: 10.0.108.137, NOT ISOLATED, Files: [], Processes: [1 process with: (Remote IP: 10.0.146.86 and Port: 22)], Scans: []
host9| IP: 10.0.146.86, NOT ISOLATED, Files: [], Processes: [], Scans: []

Your response must follow this format exactly: <action> <hostname> (e.g., action4 host4).

Best Action: 
'''
        #timestep=prompt.pop("timestep")
        #initialContext=f"You are defending a network at timestep {timestep}. Your goal is to keep the network fully functional. Choose an action from Restore, Patch, Isolate, Unisolate, Remove, Analyse. Only select actions that disrupt functionality if essential. Respond with <Action> <Hostname> only. Current network state: "
        finalContext="Respond with exactly one action in the format: <Action> <Hostname>."
        #modifiedPrompt="\n"+initialContext+"\n"+json.dumps(prompt)+"\n"+finalContext+"\n"
        modifiedPrompt=prompt
        print("Modified prompt: ", repr(modifiedPrompt))
        inputs = tokenizer(modifiedPrompt, return_tensors="pt").to(device)  # Move inputs to the same device (GPU if available)
        #print("TOKEN COUNT: ", inputs.input_ids.shape[1])
        startTime = time.time()
        with torch.no_grad():
                output = model.generate(inputs.input_ids, 
                                        use_cache=True, #Caches to avoid redundant computations - think enabled by default 
                                        early_stopping=True, #Ensure stops at EOS token (instead of continuing)
                                        max_new_tokens=25, #Number of tokens generated after the input
                                        num_beams=1, #Number of generation paths followed 
                                        #temperature=0.5, #Introduce randomness in response - way longer and not increase in quality
                                        #do_sample=True, #Similar to temperate, introduces randomness - no noticed increased in quality
                                        num_return_sequences=1, #Generate single output sequence for single input
                                        return_dict_in_generate=True, #Return generation output as a dictionary
                                        output_scores=True #Return scores for each token in the generated sequence
                                        )
        llmResponse = tokenizer.batch_decode(output.sequences, #TokenIDs generated by LLM
                                        skip_special_tokens=True, #Remove special tokens like <eos>
                                        clean_up_tokenization_spaces=False #Retain spacing as produced by model
                                        )[0]
        
        #Only keep what was generated by LLM (strip original response)
        if len(llmResponse) > len(modifiedPrompt):
                #Start at len(prompt) and go until end of string
                llmResponse=llmResponse[len(modifiedPrompt):].strip()
        
        #EXAMPLE OF GENERATED TOKENS FROM RESPONSE: ['action', '2', 'Ġhost', '4', 'Ċ', 'Explanation', ':', 'ĠĊ', 'The', 'Ġhost', '4', 'Ġhas', 'Ġsuspicious', 'Ġfiles', 'Ġand', 'Ġprocesses', ',', 'Ġso', 'Ġwe', 'Ġneed', 'Ġto', 'Ġtake', 'Ġaction', 'Ġto', 'Ġprevent']
        generated_ids = output.sequences[0][inputs.input_ids.shape[1]:]
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        #THIS IS RELYING ON FACT THAT ACTION AND HOST ARE ALWAYS FIRST 2 (REALLY 4) TOKENS
        host_idx=None
        action_idx=None
        for i, tok in enumerate(generated_tokens):
                # Match action like ['action', '2'] or ['Ġaction', '3']
                if tok.strip().lower() == 'action' and i + 1 < len(generated_tokens):
                        if generated_tokens[i + 1].isdigit():
                                action_idx = (i, i + 1)

        # Match host like ['Ġhost', '4']
        if tok.strip().strip('Ġ').lower() == 'host' and i + 1 < len(generated_tokens):
                if generated_tokens[i + 1].isdigit():
                        host_idx = (i, i + 1)
        action_idx=(0,1) if not action_idx else action_idx
        host_idx=(2,3) if not host_idx else host_idx
        
        #Get the logits for each of the first 4 tokens
        action_prefix_logits = output.scores[action_idx[0]][0]  #logits for first token, expecting "action"
        action_suffix_logits = output.scores[action_idx[1]][0]  #logits for action number: 1–7
        host_prefix_logits   = output.scores[host_idx[0]][0]  #logits for "Ġhost"
        host_suffix_logits   = output.scores[host_idx[1]][0]  #logits for host number: 1–13

        actionTemp=2
        hostTemp=4
        action_prefix_probs = torch.nn.functional.softmax(action_prefix_logits/actionTemp, dim=-1)
        action_suffix_probs = torch.nn.functional.softmax(action_suffix_logits/actionTemp, dim=-1)
        host_prefix_probs   = torch.nn.functional.softmax(host_prefix_logits/hostTemp, dim=-1)
        host_suffix_probs   = torch.nn.functional.softmax(host_suffix_logits/hostTemp, dim=-1)

        #Get token ID for 'action' and 'Ġhost'
        action_prefix_token = tokenizer("action").input_ids[0]
        host_prefix_token = tokenizer(" host").input_ids[0]  #Have to put space in front (ref ChatGPT for this one!)

        #Get token IDs for numbers 1–7 and 1–13 (action numbers, host numbers)
        action_number_tokens = {f"action{i}": tokenizer(str(i)).input_ids[0] for i in range(1, 7)}
        host_number_tokens = {f"host{i}": tokenizer(str(i)).input_ids[0] for i in range(1, 14)}
        
        #Final distributions over 7 actions and 13 hosts
        #This is done by multiplying the probability of action being the token by the corresponding action number (and same for hosts)
        action_distribution = {
        label: action_prefix_probs[action_prefix_token].item() * action_suffix_probs[token_id].item()
        for label, token_id in action_number_tokens.items()
        }

        host_distribution = {
        label: host_prefix_probs[host_prefix_token].item() * host_suffix_probs[token_id].item()
        for label, token_id in host_number_tokens.items()
}

        # Normalize action_distribution
        total_action_prob = sum(action_distribution.values())
        for k in action_distribution:
                action_distribution[k] /= total_action_prob

        # Normalize host_distribution
        total_host_prob = sum(host_distribution.values())
        for k in host_distribution:
                host_distribution[k] /= total_host_prob
                
        print("ACTION DISTRIBUTION: ", action_distribution)
        print("HOST DISTRIBUTION: ", host_distribution)
        print("LLM RESPONSE: ", llmResponse)
        

        #CREATE DISTRIBUTION FOR EVERY POSSIBLE ACTION TOO:
        jointDistribution = {}

        for action_name, p_action in action_distribution.items():
                for host_name, p_host in host_distribution.items():
                        joint_key = f"{action_name} {host_name}"
                        jointDistribution[joint_key] = p_action * p_host
        # Normalize joint_distribution
        total=sum(jointDistribution.values())
        for k in jointDistribution:
                jointDistribution[k] /= total
        # print("JOINT DISTRIBUTION: ", dict(sorted(joint_distribution.items(), key=lambda item: item[1], reverse=True)))
        # print("JOINT DISTRIBUTION: ",joint_distribution)
        
        #Change the order of the dictionary so that action3 comes after action1 (changing the order in which definitions were presented showed small increase in performance)
        #REF CHATGPT FOR THE QUICK CODE TO DO THIS DICT REORDERING :D
        custom_action_order = ['action1', 'action3', 'action2', 'action4', 'action5', 'action6', 'action7']
        action_rank = {action: i for i, action in enumerate(custom_action_order)}
        sortedJointDistribution = dict(sorted(jointDistribution.items(),key=lambda x: (action_rank[x[0].split()[0]], int(x[0].split()[1].replace('host', '')))))

        #Convert from dictionary into torch that can actually be used in RL :D
        finalDist=torch.zeros(len(sortedJointDistribution))
        for i, (k,v) in enumerate(sortedJointDistribution.items()):
                finalDist[i]=v
if __name__=="__main__":
        pass