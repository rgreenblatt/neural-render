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


def stopwatch_if(message, use):
    return stopwatch(message) if use else suppress()


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
        with stopwatch_if('list_folder', verbose):
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


def upload_file(dbx,
                fullname,
                folder,
                subfolder,
                name,
                overwrite=True,
                verbose=True):
    """Upload a file.
    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    client_modified = datetime.datetime(*time.gmtime(mtime)[:6])

    file_size = os.path.getsize(fullname)

    with stopwatch_if('upload %d bytes' % file_size, verbose):
        try:
            with open(fullname, 'rb') as f:
                CHUNK_SIZE = 16 * 1024 * 1024

                if file_size <= CHUNK_SIZE:
                    data = f.read()
                    res = dbx.files_upload(data,
                                           path,
                                           mode,
                                           client_modified=client_modified)
                else:
                    upload_session_start_result = dbx.files_upload_session_start(
                        f.read(CHUNK_SIZE))
                    cursor = dropbox.files.UploadSessionCursor(
                        session_id=upload_session_start_result.session_id,
                        offset=f.tell())
                    commit = dropbox.files.CommitInfo(
                        path=path, mode=mode, client_modified=client_modified)

                    while f.tell() < file_size:
                        if ((file_size - f.tell()) <= CHUNK_SIZE):
                            res = dbx.files_upload_session_finish(
                                f.read(CHUNK_SIZE), cursor, commit)
                        else:
                            dbx.files_upload_session_append(
                                f.read(CHUNK_SIZE), cursor.session_id,
                                cursor.offset)
                            cursor.offset = f.tell()
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
    with stopwatch_if('download', verbose):
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
