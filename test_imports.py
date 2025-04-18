#!/usr/bin/env python3
"""
Test script to check for import errors in the dags directory.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Importing homeowner_dag...")
    from dags.homeowner_dag import dag
    print("Import successful!")
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()