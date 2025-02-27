import os

# print(os.environ['HOME'])
# print(os.environ)
for key, value in os.environ.items():
    print(f'{key}: {value}')