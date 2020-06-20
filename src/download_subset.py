import sys

import dropbox

from dropbox_utils import download
from utils import mkdirs


def main():
    dbx = dropbox.Dropbox(sys.argv[1])

    mkdirs("downloads/")

    with open("all_files.txt", 'r') as f:
        lines = f.read().splitlines()

    start = 0
    end = 1000
    for i in range(start, end):
        name = lines[i]
        download(dbx, "/sphere_renders/{}".format(name),
                 "downloads/{}".format(name))


if __name__ == "__main__":
    main()
