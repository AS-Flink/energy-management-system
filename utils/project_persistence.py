import json
import os
import pandas as pd
import copy

PROJECTS_FILE = "flink_ems_projects.json"

def save_projects(projects_dict):
    """
    Saves the projects dictionary to a JSON file.
    Accepts a dictionary as input.
    """
    with open(PROJECTS_FILE, 'w') as f:
        # Deepcopy to avoid modifying the original dictionary in memory
        projects_for_save = copy.deepcopy(projects_dict)
        for proj_name, proj_data in projects_for_save.items():
            # Check if there's a DataFrame in the results to convert to JSON
            if 'results' in proj_data and isinstance(proj_data.get('results', {}).get('df'), pd.DataFrame):
                projects_for_save[proj_name]['results']['df'] = proj_data['results']['df'].to_json(orient='split')
        
        json.dump(projects_for_save, f, indent=4)

def load_projects():
    """
    Loads projects from the JSON file.
    Returns a dictionary of projects.
    """
    if os.path.exists(PROJECTS_FILE) and os.path.getsize(PROJECTS_FILE) > 0:
        try:
            with open(PROJECTS_FILE, 'r') as f:
                loaded_projects = json.load(f)
                # Convert any DataFrame JSON strings back into DataFrames
                for proj_name, proj_data in loaded_projects.items():
                    if 'results' in proj_data and isinstance(proj_data.get('results', {}).get('df'), str):
                        loaded_projects[proj_name]['results']['df'] = pd.read_json(proj_data['results']['df'], orient='split')
                return loaded_projects
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load or parse projects file. Error: {e}")
            # If the file is corrupt, return an empty dictionary
            return {}
            
    # If the file doesn't exist or is empty, return an empty dictionary
    return {}