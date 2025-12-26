import os, sys

print(f"CWD: {os.getcwd()}")
print(f"FILE: {__file__}")
print("sys.path: ")
for p in sys.path:
    print(f" {p}")
