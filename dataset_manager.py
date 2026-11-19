
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from discoveryworld.ScenarioMaker import SCENARIO_INFOS

@dataclass
class datasetManager:
    def load_dataset(force_reload: bool = False):
        dataset_path = "dataset.parquet"
        if not Path.exists(Path(dataset_path)) or force_reload:
            print("Loading dataset...")
                    
            rows = []
            for scenario, info in SCENARIO_INFOS.items():
                for difficulty in info["difficulty"]:
                    api = DiscoveryWorldAPI(threadID=1)
                    try:
                        api.loadScenario(scenario, difficulty, randomSeed=0, numUserAgents=5)
                    except ValueError:
                        api.loadScenario(scenario, difficulty, randomSeed=0, numUserAgents=1)

                    observation = api.getAgentObservation(0)
                    task = api.getTaskScorecard()[0]
                    visible = observation["ui"]["taskProgress"][0]

                    rows.append({
                        "scenarioName": scenario,
                        "difficultyStr": difficulty,
                        "maxAgents": api.numUserAgents,
                        "taskName": task["taskName"],
                        "taskDescription": visible["description"],
                        "maxScore": task["maxScore"],
                        "scorecard": [
                            {"name": item["name"], "description": item["description"], "maxScore": item["maxScore"]}
                            for item in task["scoreCard"]
                        ],
                    })
                    
    
                df = pd.DataFrame(rows).reset_index(names="id")
                df.to_parquet(dataset_path, index=False)
        else:          
            print(f"Loading dataset: {dataset_path}")
            
        df = pd.read_parquet(dataset_path)
        return df

   