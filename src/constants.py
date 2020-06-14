import os

data_path = 'data'
pickle_name = 'scenes.p'
pickle_path = os.path.join(data_path, pickle_name)
imgs_dir_name = 'imgs'


def get_img_path(index, data_path=data_path):
    return os.path.join(data_path, imgs_dir_name, 'img_{}.exr'.format(index))
