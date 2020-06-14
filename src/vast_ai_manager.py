import subprocess
import json

from generate_config import VastAIManagerConfig

def get_offers(storage, *args):
    out = subprocess.run([
        'vast', 'search', 'offers', '--raw', '--storage',
        str(storage), *args
    ],
                         capture_output=True)

    return json.loads(out.stdout)


def sort_filter_offers(offers, storage, min_inet_up, min_inet_down,
                       usable_dl_perf, min_cuda):
    out_offers = []
    for offer in offers:
        # TODO: actual profiling
        if offer['inet_up'] < min_inet_up or offer['inet_down'] < min_inet_down:
            continue
        if offer['cuda_max_good'] < min_cuda:
            continue
        storage_cost_per_hour = offer['storage_cost'] / (24 * 30)
        total_cost = offer['min_bid'] + storage_cost_per_hour * storage
        effective_dl_perf = min(offer['dlperf'], usable_dl_perf)
        dlperf_per_min_bid = effective_dl_perf / total_cost
        print(dlperf_per_min_bid)
        out_offers.append((dlperf_per_min_bid, offer))

    out_offers.sort(reverse=True)

    return out_offers

def main():
    cfg = VastAIManagerConfig()

    offers = sort_filter_offers(get_offers(cfg.storage, '--type=interruptible'),
                       cfg.storage, cfg.min_inet_up, cfg.min_inet_down,
                       cfg.usable_dl_perf, cfg.min_cuda)

    print(offers)




if __name__ == "__main__":
    main()
