import os
import sys
import shutil
import random

def main():
    if len(sys.argv) != 3:
        print("python train_test_split.py <data_dir:str> <ratio:int [0:100]>")
        return
    target_dir = sys.argv[1]
    ratio = int(sys.argv[2])
    if ratio < 0 or 100 < ratio:
        print("Ration size must be between 0 and 100")
        return
     
    target_dir = target_dir if target_dir[-1] != "/" else target_dir[:-1]
    files = os.listdir(target_dir)
    random.shuffle(files)

    ratio = int(len(files) * (ratio/100) + 0.5) 

    train_dir = target_dir + "_train/"
    test_dir = target_dir + "_test/"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    target_dir += "/"

    for file in files[:ratio]:
        source_file = target_dir + file
        shutil.copy(source_file, train_dir)

    for file in files[ratio:]:
        source_file = target_dir + file
        shutil.copy(source_file, test_dir)
    
if __name__ == '__main__':
    main()