import os
import numpy as np

def euclidean_distances(A, B):
    distance = 0
    for i in range(len(A)):
        distance += (A[i] - B[i]) ** 2
    return distance ** 0.5
    
path = "D:\label"
files = os.listdir(path)

save_path = os.getcwd()
save_path = os.path.join(save_path, 'lables')
if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in files:
    file_path = os.path.join(save_path, file)
    save_file = open(file_path, 'w')
    file_path = os.path.join(path, file)
    origin_file = open(file_path, 'r')
    lines = origin_file.readlines()
    for line in lines:
        cls, x, y, w, h = line.split()
        x, y, w, h = float(x), float(y), float(w), float(h)
        label = [cls, x, y, w, h]
        if label[0] == '0':
            top = np.array([label[1], label[2] - label[4]/3])
            mid = np.array([label[1], label[2]])
            bot = np.array([label[1], label[2] + label[4]/3])
                
        else:
            obj = np.array([label[1], label[2]])

    new_label = [0, 0, 0, 0]
    new_label[1] = euclidean_distances(obj, top) * 2
    new_label[2] = euclidean_distances(obj, mid) * 4
    new_label[3] = euclidean_distances(obj, bot) * 8
    print(new_label)
    save_file.write(' '.join(map(str, new_label)))
    save_file.close()
    origin_file.close()