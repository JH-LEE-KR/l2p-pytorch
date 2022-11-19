# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as l2p
import submitit
import datetime

def parse_args():
    parser = argparse.ArgumentParser("Submitit for multinode training L2P")

    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    else:
        raise NotImplementedError

    config_parser.add_argument("--shared_folder", type=str, default="", help="Absolute Path of shared folder for all nodes, it must be accessible from all nodes")
    config_parser.add_argument("--job_name", type=str, default="test", help="Job name")
    config_parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    config_parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    config_parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    config_parser.add_argument("--nodelist", default="ai1,ai2", type=str, help="Comma separated list of nodes to use")
    config_parser.add_argument("--gpus_per_node", default=4, type=int, help="Number of gpus to request on each node")
    config_parser.add_argument("--cpus_per_task", default=4, type=int, help="Number of CPUs to request per Task/GPU")
    config_parser.add_argument("--mem_gb", default=10, type=int, help="Memory to request for all GPUs")
    config_parser.add_argument("--partition", default="", type=str, help="Partition where to submit")
    config_parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    config_parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')

    get_args_parser(config_parser)

    return parser.parse_args()


def get_shared_folder(args) -> Path:
    if Path(args.shared_folder).is_dir():
        p = Path(args.shared_folder + f"/multinode_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    init_file = get_shared_folder(args) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file
        
class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as l2p

        self._setup_gpu_args()
        l2p.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(args) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment
    
    slurm_additional_parameters={
        "nodelist" : args.nodelist,
    }

    executor.update_parameters(
        slurm_job_name=args.job_name,
        mem_gb=args.mem_gb * args.gpus_per_node,
        gpus_per_node=args.gpus_per_node,
        tasks_per_node=args.gpus_per_node,  # one task per GPU
        slurm_cpus_per_task=args.cpus_per_task,
        nodes=args.nodes,
        timeout_min=args.timeout,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_additional_parameters=slurm_additional_parameters,
        **kwargs
    )

    executor.update_parameters(name="l2p")

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
