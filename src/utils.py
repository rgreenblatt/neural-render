from pathlib import Path
import sys


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class PrintAndLog():
    def __init__(self, file_name):
        super().__init__()

        self.file = open(file_name, 'w')
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


# Note: expects numpy format image (H x W x C)
# expects square format and clean multiple
def resize(img, output_size):
    input_size = img.shape[0]
    bin_size = input_size // output_size

    assert bin_size * output_size == input_size, "multiple must be exact"

    return img.reshape(
        (output_size, bin_size, output_size, bin_size, 3)).mean(3).mean(1)
