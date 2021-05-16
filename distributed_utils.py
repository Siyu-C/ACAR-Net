import os

import torch
import torch.distributed as dist


def get_env_variable(variables, default=None):
    for candidate in variables:
        if candidate in os.environ:
            return os.environ[candidate]
    return default


def init_distributed(local_rank, args):
    if args.nnodes is not None:
        n_nodes = args.nnodes
    else:
        n_nodes = int(get_env_variable(['SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE', 'PMI_SIZE'], default=1))
    if args.node_rank is not None:
        node_id = args.node_rank
    else:
        node_id = int(get_env_variable(['SLURM_PROCID', 'MV2_COMM_WORLD_RANK', 'PMI_RANK'], default=0))

    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['MASTER_ADDR'] = str(args.master_addr)

    world_size = n_nodes * args.nproc_per_node
    rank = node_id * args.nproc_per_node + local_rank
    dist.init_process_group(backend=args.backend, init_method='env://', world_size=world_size, rank=rank)
    print('[rank {:04d}]: distributed init: world_size={}, local_rank={}'.format(rank, world_size, local_rank), flush=True)
    
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank%num_gpus)
    
    return rank, world_size
