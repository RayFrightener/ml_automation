import importlib.util
import os
import sys

def check_file(file_path):
    try:
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return None
    except Exception as e:
        return str(e)

def check_directory(directory):
    errors = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                error = check_file(file_path)
                if error:
                    errors[file_path] = error
    return errors

if __name__ == "__main__":
    dags_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dags")
    print(f"Checking imports in {dags_dir}")
    errors = check_directory(dags_dir)
    
    if errors:
        print(f"Found {len(errors)} files with import errors:")
        for file_path, error in errors.items():
            print(f"\n{file_path}:")
            print(f"  {error}")
        sys.exit(1)
    else:
        print("No import errors found!")
        sys.exit(0)