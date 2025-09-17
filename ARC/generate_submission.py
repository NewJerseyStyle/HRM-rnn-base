import json
import torch
import numpy as np
from train_arc import ARCModel, ARCDataset, remove_padding
from torch.utils.data import DataLoader
import os


def generate_submission_with_two_attempts(model, test_loader, device, output_path):
    """
    Generate submission file with exactly 2 attempts per task output
    Following the Kaggle ARC competition format
    """
    model.eval()
    predictions = {}
    
    print("Generating predictions for test set...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            task_id = batch['task_id'][0]  # Batch size 1 for test
            input_grid = batch['input'].to(device)
            
            print(f"Processing task {i+1}: {task_id}")
            
            # Generate two different attempts
            # Attempt 1: Standard generation
            torch.manual_seed(42)  # For reproducibility
            output_grid_1 = model.generate(input_grid)
            
            # Attempt 2: Different random seed for variation
            torch.manual_seed(123)  
            output_grid_2 = model.generate(input_grid)
            
            # Convert to numpy and remove padding
            output_1 = output_grid_1[0].cpu().numpy()
            output_2 = output_grid_2[0].cpu().numpy()
            
            # Clip values to valid range (0-9 for ARC colors)
            output_1 = np.clip(output_1, 0, 9).astype(int)
            output_2 = np.clip(output_2, 0, 9).astype(int)
            
            # Remove padding to get actual grid size
            output_1 = remove_padding(output_1.tolist())
            output_2 = remove_padding(output_2.tolist())
            
            # Ensure outputs are not empty
            if not output_1 or not any(output_1):
                output_1 = [[0, 0], [0, 0]]  # Default minimal grid
            if not output_2 or not any(output_2):
                output_2 = [[0, 0], [0, 0]]  # Default minimal grid
            
            # Add to predictions dictionary
            if task_id not in predictions:
                predictions[task_id] = []
            
            predictions[task_id].append({
                "attempt_1": output_1,
                "attempt_2": output_2
            })
    
    # Ensure all tasks have predictions
    print(f"\nTotal tasks with predictions: {len(predictions)}")
    
    # Save submission file
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Submission saved to {output_path}")
    
    # Validate submission format
    validate_submission(output_path)


def validate_submission(submission_path):
    """Validate that submission follows the required format"""
    with open(submission_path, 'r') as f:
        submission = json.load(f)
    
    print("\nValidating submission format...")
    
    issues = []
    
    for task_id, outputs in submission.items():
        if not isinstance(outputs, list):
            issues.append(f"Task {task_id}: outputs should be a list")
            continue
        
        for i, output in enumerate(outputs):
            if "attempt_1" not in output:
                issues.append(f"Task {task_id}, output {i}: missing 'attempt_1'")
            elif not isinstance(output["attempt_1"], list):
                issues.append(f"Task {task_id}, output {i}: 'attempt_1' should be a 2D list")
            elif not all(isinstance(row, list) for row in output["attempt_1"]):
                issues.append(f"Task {task_id}, output {i}: 'attempt_1' should be a 2D list")
            
            if "attempt_2" not in output:
                issues.append(f"Task {task_id}, output {i}: missing 'attempt_2'")
            elif not isinstance(output["attempt_2"], list):
                issues.append(f"Task {task_id}, output {i}: 'attempt_2' should be a 2D list")
            elif not all(isinstance(row, list) for row in output["attempt_2"]):
                issues.append(f"Task {task_id}, output {i}: 'attempt_2' should be a 2D list")
    
    if issues:
        print("Validation issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("✓ Submission format is valid!")
    
    return len(issues) == 0


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths for Kaggle environment
    test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    model_path = 'best_model.pth'
    submission_path = '/kaggle/working/submission.json'
    
    # Check for local testing
    if not os.path.exists(test_path):
        test_path = 'data/arc_test.json'
        submission_path = 'submission.json'
        print("Using local paths for testing")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train_arc.py")
        return
    
    # Load test dataset
    print(f"Loading test data from {test_path}...")
    test_dataset = ARCDataset(test_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_dataset)} test examples")
    
    # Initialize and load model
    print("Loading model...")
    model = ARCModel('hrm_v2.yaml').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Generate submission
    generate_submission_with_two_attempts(model, test_loader, device, submission_path)
    
    print("\nSubmission generation complete!")


if __name__ == "__main__":
    main()