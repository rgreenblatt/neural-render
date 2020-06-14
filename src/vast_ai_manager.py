import subprocess
import json
import collections
import time

from generate_config import VastAIManagerConfig


def json_vast_command(*args):
    out = subprocess.run(['vast', *args], capture_output=True)

    return json.loads(out.stdout)


def get_offers(storage, *args):
    return json_vast_command('search', 'offers', '--raw', '--storage',
                             str(storage), *args)

def get_total_cost(offer, storage):
    return offer['min_bid'] + offer['storage_total_cost']

def sort_filter_offers(offers, storage, min_inet_up, min_inet_down,
                       usable_dl_perf, min_cuda):
    out_offers = []
    for offer in offers:
        # TODO: actual profiling
        if offer['inet_up'] < min_inet_up or offer['inet_down'] < min_inet_down:
            continue
        if offer['cuda_max_good'] < min_cuda:
            continue
        total_cost = get_total_cost(offer, storage)
        effective_dl_perf = min(offer['dlperf'], usable_dl_perf)
        dlperf_per_total_cost = effective_dl_perf / total_cost
        out_offers.append((dlperf_per_total_cost, total_cost, effective_dl_perf, offer))

    out_offers.sort(key=lambda x: x[0], reverse=True)

    return out_offers

def destroy(instance_id):
    subprocess.run(['vast', 'destroy', 'instance', str(instance_id)])

def get_instances():
    return json_vast_command('show', 'instances', '--raw')

InstanceInfo = collections.recordclass('InstanceInfo', ['dlperf_per_total_cost',
                                      'total_cost', 'offer', 'start_time',
                                      'seed'])

def main():
    cfg = VastAIManagerConfig()

    instances = {}
    instance_labels = []
    label_counter = 0

    while True:
        actual_instances = get_instances()

        for instance in actual_instances:
            if (instance['label'] is not None
                    and instance['label'] not in instance_labels):
                print("unexpected instance:", instance['label'], "taking offline")
                destroy(instance['id'])

        all_offers = []
        to_destroy_if_not_bid = []

        for instance in actual_instances:
            if (instance['next_state'] == 'stopped'
                    or instance['intended_status'] == 'stopped'):
                print("outbid instance:", instance['label'])

                all_offers.append(instance)
                to_destroy_if_not_bid.append(instance['label'])

                instance_labels.remove(instance['label'])

        all_offers.extend(get_offers(cfg.storage, '--type=interruptible'))

        offers = sort_filter_offers(all_offers, cfg.storage, cfg.min_inet_up,
                                    cfg.min_inet_down, cfg.usable_dl_perf,
                                    cfg.min_cuda)

        total_cost = sum(instances[label].total_cost
                         for label in instance_labels)
        sorted_instances = [(instances[label].dlperf_per_total_cost, label)
                            for label in instance_labels]
        sorted_instances.sort(key=lambda x: x[0])

        additional = []

        for (dlperf_per_total_cost, this_total_cost, effective_dl_perf,
             offer) in offers:
            if this_total_cost + total_cost < cfg.max_dl_per_hour:
                total_cost += this_total_cost

                is_existing = 'label' in offer
                if is_existing:
                    label = offer['label']
                else:
                    label = 'managed_instance_{}'.format(label_counter)
                    label_counter += 1
                instance_labels.append(label)
                if is_existing:
                    instances[label].total_cost = this_total_cost

                instances.append(InstanceInfo(dlperf_per_total_cost,
                                              this_total_cost,
                                              start_time=


        # def instance_start_time(i_id):
        #     return time.time() -  wl

        # replacable_instances = filter(lambda i_id: instances[i_id

        # offers = sort_filter_offers(
        #     get_offers(cfg.storage,
        #                '--type=interruptible'), cfg.storage, cfg.min_inet_up,
        #     cfg.min_inet_down, cfg.usable_dl_perf, cfg.min_cuda)

        # for per, offer in offers:
        #     print(per)

        break


if __name__ == "__main__":
    main()
