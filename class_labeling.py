import os
path = "labels/"
files = os.listdir(path)
label_file = "class_label.txt"

new_cls = []
with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            new_cls.append(line[0].strip())

idx = 0
for file in files:
    file_path = os.path.join(path, file)
    with open(file_path, 'r') as f:
        line = f.readline()
        cls, top, mid, bot = line.split()
    
    with open('new_label.txt', 'a') as f:
        f.write("%s %s %s %s\n" %(new_cls[idx], top, mid, bot))
    
    idx += 1