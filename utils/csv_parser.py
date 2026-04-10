import csv
import io
import numpy as np

def parse_csv_bytes(csv_bytes):
    """
    Converts raw bytes from the agent into a NumPy array and headers.
    """
    # Decode bytes to string
    stream = io.StringIO(csv_bytes.decode('utf-8'))
    reader = csv.reader(stream)
    
    lines = list(reader)
    if not lines:
        return None, None
    
    headers = lines[0]
    data = np.array(lines[1:]) # Everything after the header
    
    return data, headers