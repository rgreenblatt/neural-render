import subprocess
import json
import collections
import time

from generate_config import VastAIManagerConfig
from gen_utils import random_seed


def json_vast_command(*args):
    for i in range(5):
        try:
            out = subprocess.run(['vast', *args], capture_output=True)

            return json.loads(out.stdout)
        except json.decoder.JSONDecodeError as e:
            print("json err:", e)
            time.sleep(1.0)

    raise ValueError



def get_offers(storage, *args):
    return json_vast_command('search', 'offers', '--raw', '--storage',
                             str(storage), *args)


def get_total_cost(offer, storage):
    return offer['min_bid'] + offer['storage_total_cost']


def sort_filter_offers(offers, cfg):
    out_offers = []
    for offer in offers:
        # TODO: actual profiling
        if (offer['inet_up'] < cfg.min_inet_up
                or offer['inet_down'] < cfg.min_inet_down):
            continue
        if offer['cuda_max_good'] < cfg.min_cuda:
            continue

        offer['min_bid'] = max(offer['min_bid'], 0.001)
        total_cost = get_total_cost(offer, cfg.storage)
        effective_dl_perf = min(offer['dlperf'], cfg.usable_dl_perf)
        dlperf_per_total_cost = effective_dl_perf / total_cost

        if dlperf_per_total_cost < cfg.min_dlperf_per_total_cost:
            continue

        out_offers.append(
            (dlperf_per_total_cost, total_cost, effective_dl_perf, offer))

    out_offers.sort(key=lambda x: x[0], reverse=True)

    return out_offers


def destroy_instance(instance_id):
    subprocess.run(['vast', 'destroy', 'instance', str(instance_id)])


def get_instances():
    return json_vast_command('show', 'instances', '--raw')


InstanceInfo = collections.namedtuple('InstanceInfo', [
    'dlperf_per_total_cost', 'total_cost', 'offer', 'start_time', 'base_seed'
])


def set_bid(id_i, bid):
    subprocess.run(['vast', 'change', 'bid', str(id_i), '--price', str(bid)])


def create_instance(id_i, bid, label, base_seed, cfg):
    create_command = [
        'vast', 'create', 'instance',
        str(id_i), '--price',
        str(bid), '--disk',
        str(cfg.storage), '--image', cfg.image, '--label', label,
        '--onstart-cmd',
        'cd ~/neural-render && git pull && nohup python3 src/generate_upload.py ' +
        '{} --seed {} {}'.format(
            cfg.api_key, base_seed,
            cfg.base_as_arg_string() + ' &> ~/render.log &')
    ]

    print("create_command:", create_command, flush=True)
    subprocess.run(create_command)


def main():
    cfg = VastAIManagerConfig()

    instances = {}
    instance_labels = []
    label_counter = random_seed()

    while True:
        orig_actual_instances = get_instances()

        label_to_instance_id = {}
        actual_labels = []
        next_actual_instances = []
        for instance in orig_actual_instances:
            if instance['label'] is not None:
                if instance['label'] in instance_labels:
                    actual_labels.append(instance['label'])
                    next_actual_instances.append(instance)
                    label_to_instance_id[instance['label']] = instance['id']
                else:
                    print("unexpected instance:", instance['label'],
                          "- taking offline", flush=True)
                    destroy_instance(instance['id'])

        instance_labels = list(set(actual_labels))

        # list to avoid del and iter behavior
        all_keys = list(instances.keys())
        for key in all_keys:
            if key not in instance_labels:
                del instances[key]


        all_offers = []
        to_destroy_if_not_bid = []

        for instance in next_actual_instances:
            if (instance['next_state'] == 'stopped'
                    or instance['intended_status'] == 'stopped'):
                print("outbid instance:", instance['label'], flush=True)

                all_offers.append(instance)
                to_destroy_if_not_bid.append(instance['label'])

                instance_labels.remove(instance['label'])

        time.sleep(5.0)

        all_offers.extend(get_offers(cfg.storage, '--type=interruptible'))

        offers = sort_filter_offers(all_offers, cfg)

        total_cost = sum(instances[label].total_cost
                         for label in instance_labels)
        removable_instances = filter(
            lambda label: (time.time() - instances[label].start_time) > 3600 *
            cfg.min_dur_hours, instance_labels)
        sorted_instances = [(instances[label].dlperf_per_total_cost, label)
                            for label in removable_instances]
        sorted_instances.sort(key=lambda x: x[0])

        for (dlperf_per_total_cost, this_total_cost, effective_dl_perf,
             offer) in offers:
            if this_total_cost + total_cost < cfg.max_dl_per_hour:
                add_instance = True
            elif (len(sorted_instances) > 0
                  and dlperf_per_total_cost > sorted_instances[0][0]):
                label = sorted_instances[0][1]
                to_destroy_if_not_bid.append(label)
                del sorted_instances[0]
                instance_labels.remove(label)
                add_instance = True
            else:
                add_instance = False

            if add_instance:
                total_cost += this_total_cost

                is_existing = 'label' in offer

                if is_existing:
                    label = offer['label']
                else:
                    label = 'managed_instance_{}'.format(label_counter)
                    label_counter += 1

                bid_price = offer['min_bid'] * cfg.bid_multiplier
                id_i = offer['id']

                instance_labels.append(label)
                if is_existing:
                    instances[label] = instances[label]._replace(
                        total_cost=this_total_cost)
                    to_destroy_if_not_bid.remove(label)

                    print("bidding up:",
                          label,
                          "to",
                          this_total_cost,
                          flush=True)
                    set_bid(id_i, bid_price)
                else:
                    base_seed = random_seed()
                    instances[label] = InstanceInfo(dlperf_per_total_cost,
                                                    this_total_cost, offer,
                                                    time.time(), base_seed)
                    print("creating new instance:", label, flush=True)
                    create_instance(id_i, bid_price, label, base_seed, cfg)

        for label in to_destroy_if_not_bid:
            print("destroying:", label, flush=True)
            destroy_instance(label_to_instance_id[label])
            del instances[label]

        print()
        print("===== Running instances =====")
        for label in instance_labels:
            instance = instances[label]
            print(label, "at seed", instance.base_seed, "perf/cost:",
                  instance.dlperf_per_total_cost, "total cost:",
                  instance.total_cost, "gpu type:", instance.offer["gpu_name"],
                  "n gpus:", instance.offer["num_gpus"])
        print(flush=True)

        time.sleep(10.0)


if __name__ == "__main__":
    main()
