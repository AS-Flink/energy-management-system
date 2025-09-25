# utils/project_persistence.py
import json
import os
import pandas as pd
import copy

PROJECTS_FILE = "flink_ems_projects.json"

def save_projects(projects_dict):
    """Saves the projects dictionary to a JSON file."""
    with open(PROJECTS_FILE, 'w') as f:
        projects_for_save = copy.deepcopy(projects_dict)
        for proj_name, proj_data in projects_for_save.items():
            if 'results' in proj_data and isinstance(proj_data['results'].get('df'), pd.DataFrame):
                # Convert DataFrame to JSON string for saving
                projects_for_save[proj_name]['results']['df'] = proj_data['results']['df'].to_json(orient='split')
        json.dump(projects_for_save, f, indent=4)

def load_projects():
    """Loads projects from the JSON file."""
    if os.path.exists(PROJECTS_FILE) and os.path.getsize(PROJECTS_FILE) > 0:
        try:
            with open(PROJECTS_FILE, 'r') as f:
                loaded_projects = json.load(f)
                for proj_name, proj_data in loaded_projects.items():
                    if 'results' in proj_data and isinstance(proj_data['results'].get('df'), str):
                        # Convert JSON string back to DataFrame
                        proj_data['results']['df'] = pd.read_json(proj_data['results']['df'], orient='split')
                return loaded_projects
        except json.JSONDecodeError:
            print("Warning: Could not decode projects file.")
            return {}
    return {}