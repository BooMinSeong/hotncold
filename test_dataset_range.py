#!/usr/bin/env python3
"""Test script to check what happens when dataset_start/end exceed dataset size"""

from datasets import Dataset

# Create a small test dataset
test_data = {"id": list(range(10)), "text": [f"sample_{i}" for i in range(10)]}
dataset = Dataset.from_dict(test_data)

print(f"Dataset size: {len(dataset)}")
print()

# Test case 1: Normal range within bounds
print("Test 1: Normal range (0, 5)")
try:
    result = dataset.select(range(0, 5))
    print(f"✓ Success: Selected {len(result)} samples")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
print()

# Test case 2: End exceeds dataset size
print("Test 2: End exceeds size (5, 15)")
try:
    result = dataset.select(range(5, 15))
    print(f"✓ Success: Selected {len(result)} samples")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
print()

# Test case 3: Both start and end exceed dataset size
print("Test 3: Both exceed size (15, 20)")
try:
    result = dataset.select(range(15, 20))
    print(f"✓ Success: Selected {len(result)} samples")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
print()

# Test case 4: Start equals dataset size
print("Test 4: Start equals size (10, 15)")
try:
    result = dataset.select(range(10, 15))
    print(f"✓ Success: Selected {len(result)} samples")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
