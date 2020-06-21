import argparse

import dropbox

from dropbox_utils import upload_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('api_key')
    parser.add_argument('source')
    parser.add_argument('dest')
    args = parser.parse_args()

    dbx = dropbox.Dropbox(args.api_key)

    upload_file(dbx, args.source, "/", "/", args.dest)


if __name__ == "__main__":
    main()
