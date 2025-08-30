#!/usr/bin/env python3
"""
Pre-commit hook script to record file changes in JSON format.
Uses git diff to detect changes and saves them to changes/ directory.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re


def run_git_command(cmd):
  """Run a git command and return the output."""
  try:
    result = subprocess.run(
      cmd, shell=True, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    print(f"Git command failed: {cmd}")
    print(f"Error: {e.stderr}")
    return None


def get_staged_files():
  """Get list of staged files that are not in .gitignore."""
  output = run_git_command("git diff --cached --name-status")
  if not output:
    return []
  
  staged_files = []
  for line in output.split('\n'):
    if line.strip():
      parts = line.split('\t')
      if len(parts) >= 2:
        status = parts[0]
        file_path = parts[1]
        staged_files.append((status, file_path))
  
  return staged_files


def get_file_changes(file_path, status):
  """Get detailed line changes for a specific file."""
  line_changes = []
  
  if status == 'A':  # Added file
    # Get all lines of the new file
    try:
      result = subprocess.run(
        f"git show :0:{file_path}",
        shell=True, capture_output=True, text=True, check=True, encoding='utf-8'
      )
      lines = result.stdout.split('\n')
      for i, line in enumerate(lines):
        if line.strip():  # Skip empty lines
          line_changes.append({
            "line_number": i + 1,
            "change": {
              "type": "add",
              "new_line": line
            }
          })
    except subprocess.CalledProcessError:
      # File might be binary or inaccessible
      pass
      
  elif status == 'D':  # Deleted file
    # Get all lines of the deleted file
    try:
      result = subprocess.run(
        f"git show HEAD:{file_path}",
        shell=True, capture_output=True, text=True, check=True
      )
      lines = result.stdout.split('\n')
      for i, line in enumerate(lines):
        if line.strip():  # Skip empty lines
          line_changes.append({
            "line_number": i + 1,
            "change": {
              "type": "remove",
              "previous_line": line
            }
          })
    except subprocess.CalledProcessError:
      # File might be binary or inaccessible
      pass
      
  elif status == 'M':  # Modified file
    # Get unified diff for the file
    try:
      result = subprocess.run(
        f"git diff --cached -U0 {file_path}",
        shell=True, capture_output=True, text=True, check=True
      )
      diff_output = result.stdout
      
      # Parse the diff output
      current_line = 0
      for line in diff_output.split('\n'):
        # Look for hunk headers like @@ -1,3 +1,4 @@
        hunk_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if hunk_match:
          old_start = int(hunk_match.group(1))
          new_start = int(hunk_match.group(3))
          current_line = new_start
          continue
          
        if line.startswith('-') and not line.startswith('---'):
          # Removed line
          line_changes.append({
            "line_number": current_line,
            "change": {
              "type": "remove",
              "previous_line": line[1:]  # Remove the '-' prefix
            }
          })
        elif line.startswith('+') and not line.startswith('+++'):
          # Added line
          line_changes.append({
            "line_number": current_line,
            "change": {
              "type": "add",
              "new_line": line[1:]  # Remove the '+' prefix
            }
          })
          current_line += 1
        elif line.startswith(' '):
          # Context line (unchanged)
          current_line += 1
          
    except subprocess.CalledProcessError:
      # File might be binary or inaccessible
      pass
  
  return line_changes


def record_changes():
  """Main function to record all file changes."""
  try:
    # Get staged files
    staged_files = get_staged_files()
    
    print(f"DEBUG: Found {len(staged_files)} staged files: {staged_files}")
    
    if not staged_files:
      print("No staged files found.")
      return True
    
    # Process each staged file
    changed_files = []
    for status, file_path in staged_files:
      print(f"Processing {status}: {file_path}")
      
      line_changes = get_file_changes(file_path, status)
      
      if line_changes:  # Only include files with actual changes
        changed_files.append({
          "file_path": file_path,
          "line_changes": line_changes
        })
    
    if not changed_files:
      print("No meaningful changes detected in staged files.")
      return True
    
    print(f"DEBUG: About to create changes data with {len(changed_files)} files")
    
    # Create the changes data structure
    changes_data = {
      "timestamp": datetime.now().isoformat(),
      "changed_files": changed_files
    }
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    changes_file = Path("changes") / f"changes_{timestamp}.json"
    
    # Ensure changes directory exists
    changes_file.parent.mkdir(exist_ok=True)
    
    # Write the JSON file
    with open(changes_file, 'w', encoding='utf-8') as f:
      json.dump(changes_data, f, indent=2, ensure_ascii=False)
    
    print(f"Changes recorded in: {changes_file}")
    print(f"Total files processed: {len(changed_files)}")
    
    # Call main.py with --evaluate to process the changes
    try:
      print(f"Evaluating changes using main.py --evaluate...")
      eval_result = subprocess.run(
        f"python main.py -evaluate {changes_file}",
        shell=True, capture_output=True, text=True, timeout=60
      )
      
      if eval_result.returncode == 0:
        print("✅ Change evaluation completed successfully:")
        print(eval_result.stdout)
      else:
        print(f"⚠️  Change evaluation failed with exit code {eval_result.returncode}")
        print(f"Error output: {eval_result.stderr}")
        
    except subprocess.TimeoutExpired:
      print("⚠️  Change evaluation timed out after 60 seconds")
    except Exception as e:
      print(f"⚠️  Error during change evaluation: {e}")
    
    print("Continuing with commit...")
    
    return True
    
  except Exception as e:
    print(f"Warning: Failed to record changes: {e}")
    print("Continuing with commit...")
    return True  # Continue with commit even if recording fails


if __name__ == "__main__":
  success = record_changes()
