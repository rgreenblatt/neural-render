import subprocess
import os
import time
import datetime
import contextlib
import shutil

import dropbox
import numpy as np

from generate_config import GenerateUploadConfig

def upload_file(dbx, fullname, folder, subfolder, name, overwrite=False):
    """Upload a file.
    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    with stopwatch('upload %d bytes' % len(data)):
        try:
            res = dbx.files_upload(
                data, path, mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
    print('uploaded as', res.name.encode('utf8'))
    return res

def upload_dir(dbx, dirname, remote_dir_name):
    for dn, dirs, files in os.walk(dirname):
        subfolder = dn[len(dirname):].strip(os.path.sep)
        # First do all the files.
        for name in files:
            fullname = os.path.join(dn, name)
            upload_file(dbx, fullname, remote_dir_name, subfolder, name)

@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0))

if __name__ == "__main__":
    cfg = GenerateUploadConfig()

    dbx = dropbox.Dropbox(cfg.api_key)

    base_cmd = ('./scripts/blender_correct_py --background ' +
                '--python src/generate.py')
    seed = cfg.base_seed
    while True:
        os.system("{} -- {} --seed {}".format(base_cmd,
                                              cfg.base_as_arg_string(), seed))
        seed += 1

        upload_dir(dbx, "generated", cfg.app_dir)
        shutil.rmtree("generated")
