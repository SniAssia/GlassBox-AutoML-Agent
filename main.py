import time
import base64
from autofit_tool import run_autofit

def test_pipeline():
    # Load your test data
    with open('data/data.csv', 'rb') as f:
        encoded_csv = base64.b64encode(f.read()).decode('utf-8')

    start_time = time.time()
    
    # Simulate Agent Call
    response = run_autofit(encoded_csv, target_column='Class', task_type='classification')
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"--- Final Validation ---")
    print(f"Duration: {duration:.2f} seconds (Goal: < 120s)")
    print(f"Result: {response}")

if __name__ == "__main__":
    test_pipeline()