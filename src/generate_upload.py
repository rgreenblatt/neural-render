import subprocess
import os
import time
import datetime
import contextlib
import shutil

import dropbox
import numpy as np

from generate_config import GenerateUploadConfig

def upload_file(dbx, fullname, folder, subfolder, name, overwrite=True):
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
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0),
              flush=True)


if __name__ == "__main__":
    cfg = GenerateUploadConfig()

    dbx = dropbox.Dropbox(cfg.api_key)

    base_cmd = ('./scripts/blender_correct_py --background ' +
                '--python src/generate.py')
    seed_file = 'seed.out'
    if os.path.exists(seed_file):
        with open(seed_file, 'r') as f:
            seed = int(f.read())
    else:
        seed = cfg.base_seed

    while True:
        base_dir_name = 'generated'
        shutil.rmtree(base_dir_name, ignore_errors=True)
        return_code = os.system("{} -- {} --seed {}".format(
            base_cmd, cfg.base_as_arg_string(), seed))
        if return_code != 0:
            print("blender command failed")
            exit(return_code)
        dir_name = os.path.join(base_dir_name, 'seed_{}'.format(seed))
        shutil.make_archive(dir_name, 'zip', dir_name)
        shutil.rmtree(dir_name)
        for i in range(5):
            try:
                upload_dir(dbx, base_dir_name, cfg.app_dir)
                break
            except Exception as e:
                print("FAILED UPLOAD with err:", e)

        seed += 1

        with open(seed_file, 'w') as f:
            f.write(str(seed))
