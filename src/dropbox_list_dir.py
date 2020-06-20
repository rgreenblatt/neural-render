import sys

import dropbox

from dropbox_utils import list_folder


def main():
    dbx = dropbox.Dropbox(sys.argv[1])

    files = list_folder(dbx, "/", "sphere_renders")
    for data in files:
        print(data.name)


if __name__ == "__main__":
    main()
