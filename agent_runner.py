from dataclasses import dataclass
import json
import time

from agents.AgentsFactory import AgentType, AgentsFactory
from agents.base import BaseAgent, StepView
from environment.discoveryworld.discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from shared_world import randomize_start_positions, save_shared_frame

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

    def runAgent(self, agent: AgentType, scenarioName: str, difficultyStr: str, seed: int = 0, numSteps: int = 10, exportVideo: bool = False, threadId: int = 1, debug: bool = False, numUserAgents: int = 1, spawnSeed: int | None = None):
        print("Running agent '" + agent.value + "' on scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "' for " + str(numSteps) + " steps, seed " + str(seed) + ", thread ID " + str(threadId) + ", agents " + str(numUserAgents) + ".")
       
        api = DiscoveryWorldAPI(threadID=threadId)
        success = api.loadScenario(scenarioName=scenarioName, difficultyStr=difficultyStr, randomSeed=seed, numUserAgents=numUserAgents)
        if success == False:
            print("Error: Could not load scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "'.")
            return None
        if api.numUserAgents != numUserAgents:
            print("Error: '" + scenarioName + "' (" + difficultyStr + ") spawned " + str(api.numUserAgents) + " user agents but " + str(numUserAgents) + " were requested.")
            return None

        logFileSuffix = "." + scenarioName + "-" + difficultyStr + "-s" + str(seed) + "-thread" + str(api.THREAD_ID)
        agents = self._create_agents(agent, seed, api.numUserAgents)

        if spawnSeed is not None:
            randomize_start_positions(api, spawnSeed)
        save_frames = numUserAgents > 1
        if save_frames:
            save_shared_frame(api)

        startTime = time.time()
        episode = self._run_episode(api, agents, numSteps=numSteps, debug=debug, save_frames=save_frames)
        deltaTime = time.time() - startTime
        print("Elapsed time: " + str(deltaTime) + " seconds for " + str(numSteps) + " steps.")
        stepsPerSecond = numSteps / deltaTime if deltaTime else 0
        
        finalScorecard = api.getTaskScorecard()
        print("Final scorecard: ")
        print(json.dumps(finalScorecard, indent=4, sort_keys=True))

        taskName = finalScorecard[0]["taskName"]
        finalNormalizedScore = finalScorecard[0]["scoreNormalized"]
        totalScore = finalScorecard[0]["score"]
        stepsTaken = api.getStepCounter()

        print("Final normalized score for task '" + taskName + "': " + str(finalNormalizedScore))
        print("Number of steps: " + str(stepsTaken))
        print("Steps per second: " + str(stepsPerSecond))

        if exportVideo == True:
            filenameOut = "output_random_agent." + logFileSuffix + ".mp4"
            api.createAgentVideo(agentIdx=agents[0].agent_idx, filenameOut=filenameOut)

        completed = 0
        if finalScorecard[0]["completed"] == True:
            completed = 1
        completedSuccessfully = 0
        if finalScorecard[0]["completedSuccessfully"] == True:
            completedSuccessfully = 1

        return {
            "agentName": agent.value,
            "scenario": scenarioName,
            "difficulty": difficultyStr,   
            "numStepsLimit": numSteps,         
            "numUserAgents": numUserAgents,               
            "totalScore":totalScore,
            "finalNormalizedScore": finalNormalizedScore,
            "completed": completed,
            "completedSuccessfully": completedSuccessfully,   
            "stepsTaken": stepsTaken,
            "actionsTaken": episode["actionsTaken"],
            "actionsSucceeded": episode["actionsSucceeded"],
            "choosingActionTime": episode["choosingActionTime"],
            "totalTimeTaken": deltaTime,
            "stepsPerSecond": stepsPerSecond,
        }

    def _create_agents(self, agent_type: AgentType, seed: int, num_agents: int) -> list[BaseAgent]:
        return [
            AgentsFactory.create_agent(
                agent_type,
                agent_idx=agent_idx,
                seed=seed + agent_idx,
            )
            for agent_idx in range(num_agents)
        ]

    def _run_episode(self, api: DiscoveryWorldAPI, agents: list[BaseAgent], numSteps: int, debug: bool, save_frames: bool) -> dict:
        histories = {agent.agent_idx: [] for agent in agents}
        actions_taken = 0
        actions_succeeded = 0
        choosing_action_time = 0.0
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
                if observation.get("errors"):
                    raise RuntimeError(observation["errors"])
                if debug:
                    print(json.dumps(observation, indent=4, sort_keys=True))

                view = StepView(
                    observation=observation,
                    possible_actions=api.listKnownActions(limited=False),
                    teleport_locations=api.listTeleportLocationsDict(),
                    history=history,
                )
                choose_started = time.time()
                action = agent.choose_action(view)
                choosing_action_time += time.time() - choose_started
                result = api.performAgentAction(agentIdx=agent.agent_idx, actionJSON=action.copy())
                actions_taken += 1
                if isinstance(result, dict) and result.get("success") is True:
                    actions_succeeded += 1
                observation = api.getAgentObservation(agentIdx=agent.agent_idx)
                if observation.get("errors"):
                    raise RuntimeError(observation["errors"])

                message = observation["ui"].get("lastActionMessage", "")
                history.append({
                    "step": step,
                    "action": action,
                    "result": result,
                    "message": message,
                    "location": observation["ui"]["agentLocation"],
                    "inventory": observation["ui"]["inventoryObjects"],
                })
                print("Agent " + str(agent.agent_idx) + ", round " + str(step) + ": " + json.dumps(action))
                print("Success: " + str(result.get("success") if isinstance(result, dict) else result) + " | " + message)

            tick_result = api.tick()
            if not isinstance(tick_result, dict) or not tick_result.get("success"):
                raise RuntimeError(tick_result)
            if save_frames:
                save_shared_frame(api)
        return {
            "actionsTaken": actions_taken,
            "actionsSucceeded": actions_succeeded,
            "choosingActionTime": choosing_action_time,
        }
