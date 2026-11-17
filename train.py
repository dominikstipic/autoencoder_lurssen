from lurssen.dataset import RPlanDataset

img, mask, sizes, neighboorhood_graph = RPlanDataset()[-5]

for ns in neighboorhood_graph:
    print(ns)

print(sizes)