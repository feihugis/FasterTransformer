import importlib
import sys

def get_keys():
  return [k for k in dir(__builtins__) if k.startswith("__pybind11_internals_")]

mismatched = False
prev_keys = None
for m in sys.argv[1:]:
  print(f"Importing {m}...")
  importlib.import_module(m)
  current_keys = get_keys()
  print(f"  Current pybind11_internals:")
  for k in current_keys:
    print(f"    {k}")
  if prev_keys is not None and prev_keys != current_keys:
    mismatched = True
    print("ERROR: pybind11_internals version mismatched across modules")
  prev_keys = current_keys

if mismatched:
  print("Some pybind11 internals versions mismatched")
  sys.exit(1)
else:
  print("All pybind11 internals versions matched")
