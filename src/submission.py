import json
from pathlib import Path
from utils import load_arc_data
from solver import master_solver

def generate_submission(data_path, output_path):
    """
    Automates the solving process for all tasks in a dataset.
    1. Loads the challenge data.
    2. Iterates through every task.
    3. Uses master_solver to find a solution.
    4. Formats and saves the final JSON.
    """
    # Load the challenges
    data = load_arc_data(data_path)
    submission = {}

    print(f"Starting submission generation for {len(data)} tasks...")

    for task_id, task_content in data.items():
        train_examples = task_content['train']
        # Each test task can have multiple test inputs, though usually it's one
        test_inputs = task_content['test']
        
        task_predictions = []
        for test_item in test_inputs:
            prediction = master_solver(train_examples, test_item['input'])
            
            # If solver fails, provide a dummy black 1x1 grid as a placeholder
            if isinstance(prediction, str):
                prediction = [[0]]
                
            task_predictions.append({"attempt_1": prediction})
        
        submission[task_id] = task_predictions

    # Write final results to submission.json
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    # Path to the specific challenge file provided by Kaggle
    test_file = Path("../data/arc-agi_test_challenges.json")
    generate_submission(test_file, "submission.json")