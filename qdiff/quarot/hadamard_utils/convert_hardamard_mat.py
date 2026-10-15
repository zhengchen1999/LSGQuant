import os

import torch

FILE_PATHS = ["had.12.txt", "had.140.txt"]

here = os.path.dirname(os.path.abspath(__file__))

tables = {}
for file_path in FILE_PATHS:
    rows = []
    with open(os.path.join(here, file_path), 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row_values = []
            for char in line:
                if char == '+':
                    row_values.append(1)
                elif char == '-':
                    row_values.append(-1)
            rows.append(row_values)
    tensor_2d = torch.tensor(rows, dtype=torch.float32)
    tables[file_path[: -len('.txt')]] = tensor_2d

out_path = os.path.join(here, 'hadamard_mat.pth')
torch.save(tables, out_path)
print(f"saved {list(tables)} -> {out_path}")
