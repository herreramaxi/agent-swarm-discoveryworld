from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import time
from a4ai_project.agents.AgentsFactory import AgentType, AgentsFactory
from a4ai_project.agents.agentic.AgentHarness import STOP, AgentHarness
from a4ai_project.agents.baselines.base import BaseAgent, StepView
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from langchain_core.language_models.chat_models import BaseChatModel
from a4ai_project.agents.shared_model import FinishEpisode
from a4ai_project.communication.RouterSystem1 import RouterSystem1
from a4ai_project.communication.blackboard import Blackboard
from a4ai_project.environment.shared_world import randomize_start_positions, save_shared_frame

def reveal_open_dialog(api, agent_idx: int) -> None:
    """Show an open conversation in the observation.

    A JSON USE can start a dialog without filling the UI dialog box. The agent
    is then in dialog, but the observation still says is_in_dialog is false.
    """
    ui = api.ui[agent_idx]
    agent = ui.currentAgent
    if not agent.isInDialog() or ui.dialogToDisplay is not None:
        return
    other = agent.getAgentInDialogWith()
    if other is None or other.dialogTree is None:
        return
    dialog_text, options = other.dialogTree.getCurrentDialog()
    ui.dialogToDisplay = {
        "dialogText": dialog_text,
        "dialogOptions": options,
    }


@dataclass
class AgentRunner:

    def runAgent(self, agent: AgentType, scenarioName: str, difficultyStr: str, model: BaseChatModel | None = None, seed: int = 0, numSteps: int = 10, exportVideo: bool = False, threadId: int = 1, debug: bool = False, numUserAgents: int = 1, spawnSeed: int | None = None):
        print("Running agent '" + agent.value + "' on scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "' for " + str(numSteps) + " steps, seed " + str(seed) + ", thread ID " + str(threadId) + ", agents " + str(numUserAgents) + ".")

        api = DiscoveryWorldAPI(threadID=threadId)
        success = api.loadScenario(scenarioName=scenarioName, difficultyStr=difficultyStr, randomSeed=seed, numUserAgents=numUserAgents)
        if success == False:
            print("Error: Could not load scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "'.")
            return None
        if api.numUserAgents != numUserAgents:
            print("Error: '" + scenarioName + "' (" + difficultyStr + ") spawned " + str(api.numUserAgents) + " user agents but " + str(numUserAgents) + " were requested.")
            return None

        agents = self._create_agents(agent, seed, api, numSteps, model)

        if spawnSeed is not None:
            randomize_start_positions(api, spawnSeed)
        save_frames = numUserAgents > 1
        if save_frames:
            save_shared_frame(api)

        startTime = time.time()
        episode = self._run_episode(api, agents, numSteps=numSteps, debug=debug, save_frames=save_frames)
        deltaTime = time.time() - startTime
        stepsTaken = api.world.getStepCounter()
        print("Elapsed time: " + str(deltaTime) + " seconds for " + str(stepsTaken) + " steps.")
        stepsPerSecond = stepsTaken / deltaTime if deltaTime else 0

        finalScorecard = api.getTaskScorecard()
        print("Final scorecard: ")
        print(json.dumps(finalScorecard, indent=4, sort_keys=True))

        taskName = finalScorecard[0]["taskName"]
        finalNormalizedScore = finalScorecard[0]["scoreNormalized"]
        totalScore = finalScorecard[0]["score"]
       

        print("Final normalized score for task '" + taskName + "': " + str(finalNormalizedScore))
        print("Number of steps: " + str(stepsTaken))
        print("Steps per second: " + str(stepsPerSecond))

        if exportVideo == True:
            logFileSuffix = "." + scenarioName + "-" + difficultyStr + "-s" + str(seed) + "-thread" + str(api.THREAD_ID)
            filenameOut = "output_random_agent." + logFileSuffix + ".mp4"
            api.createAgentVideo(agentIdx=agents[0].agent_idx, filenameOut=filenameOut)

        completed = 0
        if finalScorecard[0]["completed"] == True:
            completed = 1
        completedSuccessfully = 0
        if finalScorecard[0]["completedSuccessfully"] == True:
            completedSuccessfully = 1

        actionsSuccessRate = episode["actionsSucceeded"]/ ( episode["actionsSucceeded"] + episode["actionsFailed"])
        
        return {
            "agentName": agent.value,
            "scenario": scenarioName,
            "difficulty": difficultyStr,
            "numStepsLimit": numSteps,
            "numUserAgents": numUserAgents,
            "totalScore": totalScore,
            "finalNormalizedScore": finalNormalizedScore,
            "completed": completed,
            "completedSuccessfully": completedSuccessfully,
            "stepsTaken": stepsTaken,
            "actionsSucceeded": episode["actionsSucceeded"],
            "actionsFailed": episode["actionsFailed"],
            "actionsSuccessRate":actionsSuccessRate,
            "totalTimeTaken": deltaTime,
            "stepsPerSecond": stepsPerSecond,
        }

    def runAgent_Agentic(self, agent: AgentType, scenarioName: str, difficultyStr: str, model: BaseChatModel, seed: int = 0, numSteps: int = 10, exportVideo: bool = False, threadId: int = 1, debug: bool = False, numUserAgents: int = 1, spawnSeed: int | None = None):
        print("Running agent '" + agent.value + "' on scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "' for " + str(numSteps) + " steps, seed " + str(seed) + ", thread ID " + str(threadId) + ", agents " + str(numUserAgents) + ".")

        api = DiscoveryWorldAPI(threadID=threadId)
        success = api.loadScenario(scenarioName=scenarioName, difficultyStr=difficultyStr, randomSeed=seed, numUserAgents=numUserAgents)
        if success == False:
            print("Error: Could not load scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "'.")
            return None
        if api.numUserAgents != numUserAgents:
            print("Error: '" + scenarioName + "' (" + difficultyStr + ") spawned " + str(api.numUserAgents) + " user agents but " + str(numUserAgents) + " were requested.")
            return None

        task_description = api.world.taskScorer.tasks[0].taskDescription   
        
        player = AgentsFactory.create_agent(
                        agent_type = agent,
                        max_num_steps = numSteps,
                        model = model,
                        agent_idx=0,
                        environment = api,
                        seed=seed    
                    )
        if spawnSeed is not None:
            randomize_start_positions(api, spawnSeed)
            
        startTime = time.time()
        response = player.invoke(task_description)
        deltaTime = time.time() - startTime
        stepsTaken = api.world.getStepCounter()
        print("Elapsed time: " + str(deltaTime) + " seconds for " + str(stepsTaken) + " steps.")
        stepsPerSecond = stepsTaken / deltaTime if deltaTime else 0
        actions_succeeded = response.get("actions_succeeded", 0)
        actions_failed = response.get("actions_failed", 0)
        finalScorecard = api.getTaskScorecard()
        print("Final scorecard: ")
        print(json.dumps(finalScorecard, indent=4, sort_keys=True))

        taskName = finalScorecard[0]["taskName"]
        finalNormalizedScore = finalScorecard[0]["scoreNormalized"]
        totalScore = finalScorecard[0]["score"]
       

        print("Final normalized score for task '" + taskName + "': " + str(finalNormalizedScore))
        print("Number of steps: " + str(stepsTaken))
        # print("Steps per second: " + str(stepsPerSecond))

        # if exportVideo == True:
        #     logFileSuffix = "." + scenarioName + "-" + difficultyStr + "-s" + str(seed) + "-thread" + str(api.THREAD_ID)
        #     filenameOut = "output_random_agent." + logFileSuffix + ".mp4"
        #     api.createAgentVideo(agentIdx=agents[0].agent_idx, filenameOut=filenameOut)

        completed = 0
        if finalScorecard[0]["completed"] == True:
            completed = 1
        completedSuccessfully = 0
        if finalScorecard[0]["completedSuccessfully"] == True:
            completedSuccessfully = 1

        actions_total = actions_succeeded + actions_failed
        actions_success_rate = actions_succeeded / actions_total if actions_total else 0

        return {
            "agentName": agent.value,
            "scenario": scenarioName,
            "difficulty": difficultyStr,
            "numStepsLimit": numSteps,
            "numUserAgents": numUserAgents,
            "totalScore": totalScore,
            "finalNormalizedScore": finalNormalizedScore,
            "completed": completed,
            "completedSuccessfully": completedSuccessfully,
            "stepsTaken": stepsTaken,
            "actionsSucceeded": actions_succeeded,
            "actionsFailed": actions_failed,
            "actionsSuccessRate": actions_success_rate,
            "totalTimeTaken": deltaTime,
            "stepsPerSecond": stepsPerSecond,
            "response": response
        }
        
    def runAgent_Agentic_blackboard(self, agent: AgentType, scenarioName: str, difficultyStr: str, model: BaseChatModel, seed: int = 0, numSteps: int = 10, exportVideo: bool = False, threadId: int = 1, debug: bool = False, numUserAgents: int = 1, spawnSeed: int | None = None):
            print("Running agent '" + agent.value + "' on scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "' for " + str(numSteps) + " steps, seed " + str(seed) + ", thread ID " + str(threadId) + ", agents " + str(numUserAgents) + ".")
    
            api = DiscoveryWorldAPI(threadID=threadId)
            success = api.loadScenario(scenarioName=scenarioName, difficultyStr=difficultyStr, randomSeed=seed, numUserAgents=numUserAgents)
            if success == False:
                print("Error: Could not load scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "'.")
                return None
            if api.numUserAgents != numUserAgents:
                print("Error: '" + scenarioName + "' (" + difficultyStr + ") spawned " + str(api.numUserAgents) + " user agents but " + str(numUserAgents) + " were requested.")
                return None
            if spawnSeed is not None:
                randomize_start_positions(api, spawnSeed)
                            
            task_description = api.world.taskScorer.tasks[0].taskDescription   
                        
            # players = self._create_agents(agent_type=AgentType.BlackboardAgenticPlayer, seed = seed, api= api, numSteps=numSteps,model=model)
            blackboard = Blackboard(router=RouterSystem1())
            print("Blackboard database: " + str(blackboard.db_path.resolve()))
            
            harnesses = {}
            for  idx in range(numUserAgents):
                name = f"agent-{idx}"
                harness = AgentHarness(
                    agent_idx=idx,
                    agent_nickname= name,
                    model = model,
                    environment=api,
                    max_num_steps=numSteps,
                    blackboard = blackboard             
                )                
                harnesses[name]= harness
        
            startTime = time.time()
            # response = player.invoke(task_description)
            
            blackboard.broadcast_message("scenario", task_description)
            finished = None
            finished_by = None
            with ThreadPoolExecutor(max_workers=len(harnesses)) as pool:
                try:
                    futures = {
                        pool.submit(harness.invoke): harness
                        for harness in harnesses.values()
                    }

                    for future in as_completed(futures):
                        outcome = future.result()
                        if not isinstance(outcome, dict):
                            continue
                        structured = outcome.get("structured_response")
                        if finished is None and isinstance(structured, FinishEpisode):
                            finished = structured
                            finished_by = futures[future].agent_nickname
                            for harness in harnesses.values():
                                harness.queue.put(STOP)
                except BaseException:
                    
                    for harness in harnesses.values():
                        harness.queue.put(STOP)
                    raise

            if finished is None:
                raise RuntimeError("All harnesses exited without a FinishEpisode.")

            actions_succeeded = sum(harness.agent.actions_succeeded for harness in harnesses.values())
            actions_failed = sum(harness.agent.actions_failed for harness in harnesses.values())
            response = {
                "finished_by": finished_by,
                **finished.model_dump(),
                "actions_succeeded": actions_succeeded,
                "actions_failed": actions_failed,
            }
                        
            deltaTime = time.time() - startTime
            stepsTaken = api.world.getStepCounter()
            print("Elapsed time: " + str(deltaTime) + " seconds for " + str(stepsTaken) + " steps.")
            stepsPerSecond = stepsTaken / deltaTime if deltaTime else 0
            actions_succeeded = response.get("actions_succeeded", 0)
            actions_failed = response.get("actions_failed", 0)
            finalScorecard = api.getTaskScorecard()
            print("Final scorecard: ")
            print(json.dumps(finalScorecard, indent=4, sort_keys=True))
    
            taskName = finalScorecard[0]["taskName"]
            finalNormalizedScore = finalScorecard[0]["scoreNormalized"]
            totalScore = finalScorecard[0]["score"]
            
    
            print("Final normalized score for task '" + taskName + "': " + str(finalNormalizedScore))
            print("Number of steps: " + str(stepsTaken))
            # print("Steps per second: " + str(stepsPerSecond))
    
            # if exportVideo == True:
            #     logFileSuffix = "." + scenarioName + "-" + difficultyStr + "-s" + str(seed) + "-thread" + str(api.THREAD_ID)
            #     filenameOut = "output_random_agent." + logFileSuffix + ".mp4"
            #     api.createAgentVideo(agentIdx=agents[0].agent_idx, filenameOut=filenameOut)
    
            completed = 0
            if finalScorecard[0]["completed"] == True:
                completed = 1
            completedSuccessfully = 0
            if finalScorecard[0]["completedSuccessfully"] == True:
                completedSuccessfully = 1
    
            actions_total = actions_succeeded + actions_failed
            actions_success_rate = actions_succeeded / actions_total if actions_total else 0
    
            return {
                "agentName": agent.value,
                "scenario": scenarioName,
                "difficulty": difficultyStr,
                "numStepsLimit": numSteps,
                "numUserAgents": numUserAgents,
                "totalScore": totalScore,
                "finalNormalizedScore": finalNormalizedScore,
                "completed": completed,
                "completedSuccessfully": completedSuccessfully,
                "stepsTaken": stepsTaken,
                "actionsSucceeded": actions_succeeded,
                "actionsFailed": actions_failed,
                "actionsSuccessRate": actions_success_rate,
                "totalTimeTaken": deltaTime,
                "stepsPerSecond": stepsPerSecond,
                "response": response,
                "invocationTrace": sorted(
                    (record for harness in harnesses.values() for record in harness.invocation_trace),
                    key=lambda record: record["started_at"],
                ),
            }

    def _create_agents(self, agent_type: AgentType, seed: int, api: DiscoveryWorldAPI, numSteps: int, model: BaseChatModel | None) -> list[BaseAgent]:
        num_agents = api.numUserAgents
        possible_actions=api.listKnownActions(limited=False)
        teleport_locations=api.listTeleportLocationsDict()
        task_description = api.world.taskScorer.tasks[0].taskDescription
                 
        return [
            AgentsFactory.create_agent(
                agent_type,
                agent_idx=agent_idx,
                agent_nickname = f"agent-{agent_idx}",
                seed=seed + agent_idx,          
                task_description = task_description,
                possible_actions = possible_actions,
                teleport_locations = teleport_locations,
                max_num_steps = numSteps,
                model = model
            )
            for agent_idx in range(num_agents)
        ]

    def _run_episode(self, api: DiscoveryWorldAPI, agents: list[BaseAgent], numSteps: int, debug: bool, save_frames: bool) -> dict:
        histories = {agent.agent_idx: [] for agent in agents}
        actions_failed = 0
        actions_succeeded = 0
        for step in range(numSteps):
            if api.areTasksComplete():
                break

            print("\n\n")
            print("-----------------------------------------------------------")
            print("Step " + str(step) + " of " + str(numSteps))
            print("-----------------------------------------------------------")
            print("")
            taskScorecard = api.getTaskScorecard()
            if debug:
                print("Task Scorecard: ")
                print(json.dumps(taskScorecard, indent=4, sort_keys=True))
            else:
                print("Task Score (normalized): " + str(taskScorecard[0]["scoreNormalized"]))

            for agent in agents:
                history = histories[agent.agent_idx]
                reveal_open_dialog(api, agent.agent_idx)
                observation = api.getAgentObservation(agentIdx=agent.agent_idx)                
              
                # with open(f"observation-{step}.json", "w") as f:
                #     json.dump(observation, f, indent=4, sort_keys=True)
                    
                # with open(f"history-{step}.json", "w") as f:
                #     json.dump(history, f, indent=4, sort_keys=True)                    
                        
                if observation.get("errors"):
                    raise RuntimeError(observation["errors"])
                if debug:
                    print(json.dumps(observation, indent=4, sort_keys=True))

                view = StepView(
                    observation=observation,                    
                    history=history,
                )
                
                action = agent.choose_action(view)
                result = api.performAgentAction(agentIdx=agent.agent_idx, actionJSON=action.copy())
                
                if result.get("success") is True:
                    actions_succeeded += 1
                else:
                    actions_failed += 1
                    
                # observation = api.getAgentObservation(agentIdx=agent.agent_idx)
                # if observation.get("errors"):
                #     raise RuntimeError(observation["errors"])

                # message = observation["ui"].get("lastActionMessage", "")
                history.append({
                    "step": step,
                    "action": action,
                    "result": result,
                    # "message": message,
                    # "location": observation["ui"]["agentLocation"],
                    # "inventory": observation["ui"]["inventoryObjects"],
                })
                
                # with open(f"history-post-action-{step}.json", "w") as f:
                #     json.dump(history[-1], f, indent=4, sort_keys=True)                   
                                    
                print("Agent " + str(agent.agent_idx) + ", round " + str(step) + ": " + json.dumps(action))
                # print("Success: " + str(result.get("success") if isinstance(result, dict) else result) + " | " + message)
                print("Success: " + str(result.get("success") if isinstance(result, dict) else result))

            tick_result = api.tick()
            if not isinstance(tick_result, dict) or not tick_result.get("success"):
                raise RuntimeError(tick_result)
            if save_frames:
                save_shared_frame(api)
        return {  
            "actionsSucceeded": actions_succeeded,
            "actionsFailed": actions_failed
        }
