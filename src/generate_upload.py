import os
import shutil

import dropbox

from generate_config import GenerateUploadConfig

from dropbox_utils import upload_dir


def main():
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


if __name__ == "__main__":
    main()
