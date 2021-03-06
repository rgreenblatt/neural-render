import os
import argparse
import sys
import datetime
import math
import itertools

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import make_grid
from scipy.spatial.transform import Rotation as R
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from apex import amp, optimizers
import git

from arch import net_params
from config import Config
from constants import pickle_path, get_img_path
from criterion import PerceptualLoss
from load_data import DatasetManager
from model import Net
from torch_utils import (LRSched, linear_to_srgb, LossTracker, ImageTracker,
                         EMATracker)
from utils import mkdirs, PrintAndLog


def main():
    cfg = Config()

    disable_all_output = cfg.local_rank != 0 or cfg.profile

    if not disable_all_output:
        output_dir = os.path.join("outputs", cfg.name)
        tensorboard_output = os.path.join(output_dir, "tensorboard")
        model_save_output = os.path.join(output_dir, "models")

        if os.path.exists(output_dir):
            print("output directory exists, returning")
            sys.exit(1)

        mkdirs(tensorboard_output)
        mkdirs(model_save_output)

        writer = SummaryWriter(log_dir=tensorboard_output)

        logger = PrintAndLog(os.path.join(output_dir, "output.log"))

        cfg.print_params()
        cfg.print_non_default()

        writer.add_text("params", cfg.as_markdown())
        writer.add_text("non default params", cfg.non_default_as_markdown())

        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
            writer.add_text("git commit hash", commit_hash)
            print("git commit hash:", commit_hash)
        except Exception as e:
            print("failed to get git commit hash with err:")
            print(e)

        print()

    torch.backends.cudnn.benchmark = not cfg.no_cudnn_benchmark

    use_distributed = False
    if 'WORLD_SIZE' in os.environ:
        use_distributed = int(os.environ['WORLD_SIZE']) > 1

    which_gpu = 0
    world_size = 1

    if use_distributed:
        which_gpu = cfg.local_rank
        torch.cuda.set_device(which_gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_size = torch.distributed.get_world_size()

    def reduce_tensor(tensor):
        if not use_distributed:
            return tensor

        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= world_size

        return rt

    # This is redundant I think
    device = torch.device(
        "cuda:{}".format(which_gpu) if torch.cuda.is_available() else "cpu")

    batch_size = cfg.batch_size
    valid_prop = 0.2
    max_valid_size = 20000

    show_model_info = cfg.show_model_info and not disable_all_output

    img_width = cfg.resolution

    # function for processing the input (shouldn't have trainable params)
    # takes np array
    def process_input(inp):
        # add many angle representations (not sure which is best yet)

        # switch from (W, X, Y, Z) to (X, Y, Z, W)
        quat_arr = np.concatenate((inp[:, 4:7], inp[:, 3:4]), axis=1)
        rot = R.from_quat(quat_arr)
        mat_arr = rot.as_matrix()
        mat_arr.resize((inp.shape[0], 9))
        rot_vec_arr = rot.as_rotvec()

        location = inp[:, :3]
        scale = inp[:, 7:10]

        cube_points = np.empty((inp.shape[0], 24))
        for i, point in enumerate(
                itertools.product([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])):
            point = np.tile(np.array(point), (inp.shape[0], 1))

            point *= scale
            point = rot.apply(point)
            point += location

            cube_points[:, i * 3:(i + 1) * 3] = point

        return np.concatenate((inp, mat_arr, rot_vec_arr, cube_points), axis=1)

    input_size = 56  # 20, then 56 after process_input

    blocks_args, global_args = net_params(input_size, img_width, cfg)

    net = Net(blocks_args, global_args)

    # does the order of these matter???
    if use_distributed and not cfg.no_sync_bn:
        net = convert_syncbn_model(net)

    if torch.cuda.is_available() and not cfg.no_fused_adam:
        Adam = optimizers.FusedAdam
    else:
        Adam = torch.optim.Adam

    optimizer = Adam(net.parameters(), lr=0.1, weight_decay=0.0)

    net = net.to(device)

    net, optimizer = amp.initialize(net,
                                    optimizer,
                                    opt_level=cfg.opt_level,
                                    min_loss_scale=65536.0,
                                    verbosity=cfg.amp_verbosity)

    if use_distributed:
        net = DistributedDataParallel(net)

    if cfg.no_perceptual_loss:
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = PerceptualLoss().to(device)

    if show_model_info:
        print(net)

    def recursive_param_print(module, memo=None, value='', name="net"):
        if memo is None:
            memo = set()

        total = 0
        to_print = ""

        if module not in memo:
            memo.add(module)

            for m_name, c_module in module.named_children():
                child_total, child_to_print = recursive_param_print(
                    c_module, memo, "  " + value, name=m_name)

                total += child_total
                to_print += child_to_print

            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    this_param_total = param.numel()
                    total += this_param_total
                    to_print += "  " + value + p_name + " (param) : {}\n".format(
                        this_param_total)

            if total != 0:
                to_print = value + name + ": {}\n".format(total) + to_print

        return total, to_print

    if show_model_info:
        print()

        total, to_print = recursive_param_print(net)
        print(to_print)

    train_batches_save = math.ceil(cfg.train_images_to_save / batch_size)
    test_batches_save = math.ceil(cfg.test_images_to_save / batch_size)

    if not disable_all_output:
        print("getting dataset...")

    # this is somewhat expensive, so we don't do so by default
    compute_overall_stats = False

    dataset_m = DatasetManager(pickle_path,
                               get_img_path,
                               img_width,
                               batch_size,
                               valid_prop,
                               max_valid_size,
                               cfg.valid_split_seed,
                               num_workers=8,
                               fake_data=cfg.fake_data,
                               process_input=process_input,
                               data_count_limit=cfg.data_count_limit,
                               num_replicas=world_size,
                               rank=cfg.local_rank,
                               compute_overall_stats=compute_overall_stats)

    if not disable_all_output:
        print("finished loading dataset")

    max_seq_len = None
    min_prop_emissive = None
    increase_max_seq_len = cfg.start_max_seq_len is not None
    decrease_min_prop_emissive = cfg.start_min_prop_emissive is not None
    change_factors = increase_max_seq_len or decrease_min_prop_emissive
    if increase_max_seq_len:
        max_seq_len = cfg.start_max_seq_len
    if increase_max_seq_len:
        min_prop_emissive = cfg.start_min_prop_emissive

    if not disable_all_output:
        print("getting current train test split...")

    train, test, epoch_callback = dataset_m.get_train_test(
        max_seq_len, min_prop_emissive)

    if not disable_all_output:
        print("finished getting current train test split")

    if not disable_all_output and compute_overall_stats:
        print("overall max seq len:", dataset_m.overall_max_seq_len)
        print("overall avg seq len:", dataset_m.overall_avg_seq_len)
        print("overall max prop emissive:", dataset_m.max_prop_emissive)
        print("overall min prop emissive:", dataset_m.min_prop_emissive)
        print("overall avg prop emissive:", dataset_m.avg_prop_emissive)

    step = 0

    if not disable_all_output:
        print()
        print("===== Training =====")
        print()

    factor = 2e-6
    world_batch_size = batch_size * world_size

    steps_per_epoch = len(train) * world_batch_size

    scaled_lr = cfg.lr_multiplier * world_batch_size * factor
    lr_schedule = LRSched(scaled_lr, cfg.epochs * steps_per_epoch)

    lr = 0

    norm_avg = EMATracker(0.9, 50.0)

    for epoch in range(cfg.epochs):
        if change_factors and (epoch % cfg.change_factors_freq) == 0:
            if epoch != 0:
                if max_seq_len is not None:
                    max_seq_len *= 2
                    if max_seq_len > dataset_m.overall_max_seq_len:
                        max_seq_len = None
                if min_prop_emissive is not None:
                    min_prop_emissive -= cfg.min_prop_emissive_change_rate
                    if min_prop_emissive <= 0.0:
                        min_prop_emissive = None

            train, test, epoch_callback = dataset_m.get_train_test(
                max_seq_len, min_prop_emissive)

            steps_per_epoch = len(train) * world_batch_size

            if max_seq_len is None and min_prop_emissive is None:
                # main lr schedule
                lr_schedule = LRSched(scaled_lr,
                                      (cfg.epochs - epoch) * steps_per_epoch,
                                      start_div_factor=16.0,
                                      offset=epoch)
                change_factors = False
            else:
                # change_factors lr schedule
                lr_schedule = LRSched(scaled_lr,
                                      cfg.change_factors_freq *
                                      steps_per_epoch,
                                      start_div_factor=16.0,
                                      pct_start=1.0,
                                      offset=epoch)

            if not disable_all_output:
                print("max seq len {}, min prop emissive {}, train size {}".
                      format(max_seq_len, min_prop_emissive,
                             len(train) * world_batch_size))
                increase_seq = False

        epoch_callback(epoch)

        net.train()

        i = 0

        train_loss_tracker = LossTracker(reduce_tensor)

        steps_since_display = 0
        max_train_step = (len(train) - 1) * world_batch_size
        format_len = math.floor(math.log10(max_train_step)) + 1

        def train_display():
            train_loss, nan_count = train_loss_tracker.query_reset()
            if not disable_all_output:
                print("{}, epoch {}/{}, step {}/{}, train loss {:.4e}, NaN {}".
                      format(datetime.datetime.now(), epoch, cfg.epochs - 1,
                             str(i * world_batch_size).zfill(format_len),
                             max_train_step, train_loss,
                             nan_count * world_batch_size),
                      flush=True)
                writer.add_scalar("loss/train", train_loss, step)
                writer.add_scalar("lr", lr, step)
                writer.flush()

        if not disable_all_output:
            actual_images_train = ImageTracker()
            output_images_train = ImageTracker()

        steps_since_set_lr = 0

        def reset_bn():
            if use_distributed:
                net.module.reset_running_stats()
            else:
                net.reset_running_stats()

        def set_lr():
            nonlocal lr
            lr = lr_schedule(step)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        set_lr()

        def evaluate_on_data(data):
            image = data['image'].to(device)
            inp = data['inp'].to(device)
            mask = data['mask'].to(device)
            count = data['count'].to(device)

            outputs = net(inp, mask, count)

            loss = criterion(outputs, image)

            return outputs, loss, image

        for i, data in enumerate(train):
            # this wouldn't be strictly accurate if we had partial batches
            step += world_batch_size
            steps_since_display += world_batch_size
            steps_since_set_lr += world_batch_size

            optimizer.zero_grad()

            outputs, loss, image = evaluate_on_data(data)

            # save at end of training
            if not disable_all_output and len(train) - i <= train_batches_save:
                actual_images_train.update(image)
                output_images_train.update(outputs)

            is_nan = train_loss_tracker.update(loss)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if is_nan:
                reset_bn()
            else:
                max_norm = norm_avg.x * 1.5
                this_norm = nn.utils.clip_grad_norm_(net.parameters(),
                                                     max_norm)
                norm_avg.update(this_norm)
                if not disable_all_output:
                    writer.add_scalar("max_norm", max_norm, step)
                    writer.add_scalar("norm", this_norm, step)

                optimizer.step()

            if cfg.profile and step >= cfg.profile_len:
                return

            if steps_since_set_lr >= cfg.set_lr_freq:
                set_lr()
                steps_since_set_lr = 0

            if steps_since_display >= cfg.display_freq:
                train_display()
                steps_since_display = 0

        if steps_since_display > 0:
            train_display()

        net.eval()

        test_loss_tracker = LossTracker(reduce_tensor)

        actual_images_test = ImageTracker()
        output_images_test = ImageTracker()

        with torch.no_grad():
            for i, data in enumerate(test):
                outputs, loss, image = evaluate_on_data(data)

                if not disable_all_output and i < test_batches_save:
                    actual_images_test.update(image)
                    output_images_test.update(outputs)

                test_loss_tracker.update(loss)

        test_loss, count_nan = test_loss_tracker.query_reset()
        if not disable_all_output:
            print(
                "{}, epoch {}/{}, lr {:.4e}, test loss {:.4e}, NaN {}".format(
                    datetime.datetime.now(), epoch, cfg.epochs - 1, lr,
                    test_loss, count_nan * world_batch_size),
                flush=True)

            actual_images_train = actual_images_train.query_reset()
            output_images_train = output_images_train.query_reset()
            actual_images_test = actual_images_test.query_reset()
            output_images_test = output_images_test.query_reset()

            if actual_images_train is not None:
                writer.add_image("images/train/actual",
                                 make_grid(actual_images_train), step)
                writer.add_image("images/train/output",
                                 make_grid(output_images_train), step)

            if actual_images_test is not None:
                writer.add_image("images/test/actual",
                                 make_grid(actual_images_test), step)
                writer.add_image("images/test/output",
                                 make_grid(output_images_test), step)

            writer.add_scalar("loss/test", test_loss, step)

        # if not disable_all_output and (epoch + 1) % cfg.save_model_every == 0:
        #     torch.save(
        #         net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    # if not disable_all_output:
    #     torch.save(net, os.path.join(model_save_output, "net_final.p"))


if __name__ == "__main__":
    main()
