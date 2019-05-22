import sys
import pickle
import glob

def score(filename):
    with open(filename, mode="rb") as f:
        data = pickle.load(f)
    acc = 0
    for rewards in data[-1]:
        acc += max(rewards)
    return acc

if __name__ == "__main__":
    files = glob.glob(sys.argv[1] + "*")
    for file in files:
        print("file: ", file)
        try:
            print(score(file))
        except:
            pass
