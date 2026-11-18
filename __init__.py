

import argparse
import datetime
import json
import random 
import time

from agents.AgentsFactory import AgentConfiguration, AgentType
from agent_runner import AgentRunner
from environment.discoveryworld.discoveryworld.ScenarioMaker import SCENARIO_DIFFICULTY_OPTIONS, SCENARIO_NAMES, getInternalScenarioName


def main():
    print("Initializing DiscoveryWorld API... ")

    # Randomly generate a thread ID, in case one isn't specified
    # Random seed based on the current time
    rThread = random.Random()
    rThread.seed(int(time.time()))
    randomThreadId = rThread.randint(1, 10000000)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Play DiscoveryWorld using Random Baseline Agent.")
    parser.add_argument(
        '--agent',
        choices=[agent_type.value for agent_type in AgentType],
        default=AgentType.Random.value,
    )
    parser.add_argument('--scenario', choices=SCENARIO_NAMES, default=None)
    parser.add_argument('--difficulty', choices=list(SCENARIO_DIFFICULTY_OPTIONS.values()), default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--numSteps', type=int, default=100)
    parser.add_argument('--numAgents', type=int, default=1, help='User agents in the shared world (default: 1)')
    parser.add_argument('--spawnSeed', type=int, default=None, help='Place agents on distinct landmarks. Omit to keep scenario spawns.')
    parser.add_argument('--runall', action='store_true', help='Run all scenarios with random agent')
    parser.add_argument('--video', action='store_true', help='Export video of agent actions')
    parser.add_argument('--threadId', type=int, default=randomThreadId)

    args = parser.parse_args()
    agent_config = AgentConfiguration(agent_type=AgentType(args.agent))
    print("Using Agent: " + agent_config.agent_type.value)
    print("Using Thread ID: " + str(args.threadId))
    print("This can be specified with the '--threadId' argument.")
    time.sleep(1)

    # Timestamp
    now = datetime.datetime.now()
    dateStr = now.strftime("%Y-%m-%d_%H-%M-%S")

    if (args.scenario == None):
        print("Error: Must specify a scenario (or use --runall to run all scenarios).")
        print("Available scenarios: " + str(SCENARIO_NAMES))
        raise SystemExit(1)
    if (args.difficulty == None):
        print("Error: Must specify a difficulty.")
        print("Available difficulties: " + str(list(SCENARIO_DIFFICULTY_OPTIONS.values())))
        raise SystemExit(1)

    # Get the internal scenario name
    internalScenarioName = getInternalScenarioName(args.scenario, args.difficulty)
    if (internalScenarioName == None):
        print("Error: Could not find internal scenario name for scenario '" + args.scenario + "' and difficulty '" + args.difficulty + "'.")
        print("Available scenarios: " + str(SCENARIO_NAMES))
        print("Available difficulties: " + str(list(SCENARIO_DIFFICULTY_OPTIONS.values())))
        raise SystemExit(1)

    threadId = args.threadId
    finalScores = AgentRunner().runAgent(agent=agent_config.agent_type, scenarioName=args.scenario, difficultyStr=args.difficulty, seed=args.seed, numSteps=args.numSteps, exportVideo=args.video, threadId=threadId, debug=False, numUserAgents=args.numAgents, spawnSeed=args.spawnSeed)
    if (finalScores == None):
        print("Run failed for '" + args.scenario + "' / '" + args.difficulty + "'.")
        raise SystemExit(1)

    # finalScores["seed"] = args.seed
    # finalScores["threadId"] = threadId

    print("Final scores: ")
    print(json.dumps(finalScores, indent=4, sort_keys=True))
    filenameOut = safeFilename("output_random_agent." + args.scenario + "-" + args.difficulty + "-s" + str(args.seed) + "-thread" + str(threadId) + "." + dateStr + ".json")
    print("Writing " + filenameOut + "...")
    with open(filenameOut, "w") as f:
        json.dump(finalScores, f, indent=4, sort_keys=True)
    print("Completed...")


def scenarioPairs(runall:bool, scenario:str, difficulty:str):
    if (not runall):
        return [(scenario, difficulty)]

    pairs = []
    for name in SCENARIO_NAMES:
        for diff in SCENARIO_DIFFICULTY_OPTIONS.values():
            if (getInternalScenarioName(name, diff) != None):
                pairs.append((name, diff))
    return pairs

def safeFilename(name:str):
    # Windows filenames cannot contain these characters. Scenario names include ":".
    return "".join("-" if ch in '<>:"/\\|?*' else ch for ch in name)

if __name__ == "__main__":
    main()
