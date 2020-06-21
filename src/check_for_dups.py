import pickle

from constants import pickle_path

from tqdm import tqdm


def main():
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    seen = set()
    dup = []
    for i, x in tqdm(enumerate(data)):
        x = tuple(x.flatten().tolist())
        if x in seen:
            dup.append((i, x))
        seen.add(x)

    print(dup)

if __name__ == "__main__":
    main()
