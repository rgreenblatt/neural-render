import contextlib
from contextlib import suppress
import datetime
import os
import time

import dropbox


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


def list_folder(dbx, folder, subfolder, verbose=True):
    """List a folder.
    Return a dict mapping unicode filenames to
    FileMetadata|FolderMetadata entries.
    """
    path = '/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'))
    while '//' in path:
        path = path.replace('//', '/')
    path = path.rstrip('/')
    entries = []
    try:
        context = stopwatch('list_folder') if verbose else suppress()
        with context:
            res = dbx.files_list_folder(path)
            entries.extend(res.entries)
            has_more = res.has_more
            while has_more:
                res = dbx.files_list_folder_continue(res.cursor)
                entries.extend(res.entries)
                has_more = res.has_more

    except dropbox.exceptions.ApiError as err:
        print('Folder listing failed for', path, 'err:', err)
        return []
    else:
        return entries


def upload_file(dbx, fullname, folder, subfolder, name, overwrite=True, verbose=True):
    """Upload a file.
    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    context = stopwatch('upload %d bytes' %
                        len(data)) if verbose else suppress()
    with context:
        try:
            res = dbx.files_upload(
                data,
                path,
                mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
    if verbose:
        print('uploaded as', res.name.encode('utf8'))
    return res


def upload_dir(dbx, dirname, remote_dir_name):
    for dn, dirs, files in os.walk(dirname):
        subfolder = dn[len(dirname):].strip(os.path.sep)
        # First do all the files.
        for name in files:
            fullname = os.path.join(dn, name)
            upload_file(dbx, fullname, remote_dir_name, subfolder, name)


def download(dbx, path, local_path, verbose=True):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    while '//' in path:
        path = path.replace('//', '/')
    context = stopwatch('download') if verbose else suppress()
    with context:
        try:
            md, res = dbx.files_download(path)
        except dropbox.exceptions.HttpError as err:
            print('*** HTTP error', err)
            return None
    data = res.content
    if verbose:
        print(len(data), 'bytes; md:', md)

    with open(local_path, 'wb') as f:
        f.write(data)

    return data
