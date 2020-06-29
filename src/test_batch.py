from itertools import chain, combinations

import torch
from tqdm import tqdm

from arch import net_params
from model import Net
from config import Config
from load_data import mask_collate_fn

def main():
    with torch.no_grad():
        cfg = Config()

        input_size = 5
        img_width = 8

        blocks_args, global_args = net_params(input_size, img_width, cfg)

        net = Net(blocks_args, global_args)

        net.eval()

        inputs = [
            torch.randn(2, input_size),
            torch.randn(1, input_size),
            torch.randn(8, input_size),
            torch.randn(18, input_size),
            torch.randn(12, input_size),
        ]
        inputs = list(
            enumerate(map(lambda x: {
                'image': torch.tensor(1.0),
                'inp': x
            }, inputs)))

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)

            return chain.from_iterable(
                combinations(s, r) for r in range(len(s) + 1))

        results = [[] for _ in inputs]

        for comb in tqdm(powerset(inputs)):
            if len(comb) == 0:
                continue

            indexes = list(map(lambda x : x[0], comb))
            values = list(map(lambda x : x[1], comb))

            data = mask_collate_fn(values)
            inp = data['inp']
            mask = data['mask']
            count = data['count']

            outputs = net(inp, mask, count)

            assert outputs.size(0) == len(indexes)

            for i, index in enumerate(indexes):
                results[index].append(outputs[i])

            outputs_later = net(inp, mask, count)

            assert (outputs == outputs_later).all()

        delta = 1e-5
        for result in results:
            for i in range(1, len(result)):
                l1 = (result[0] - result[i]).abs().mean()
                assert l1 < delta, "failed at {}, l1 {}".format(i, l1)
        for i in range(1, len(results)):
            l1 = (results[0][0] - results[i][0]).abs().mean()
            assert l1 > delta, "failed at {}, l1 {}".format(i, l1)


if __name__ == "__main__":
    main()
