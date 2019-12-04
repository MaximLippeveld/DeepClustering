import argparse
from func import augtest, lmdbtest
from func import fit_dae, fit_cae
from func import dyn_ae_clustering
from func import mc_dropout_cae
import os
import importlib
from pathlib import Path
import shutil
import json
import torch
import numpy
import torch.multiprocessing
import sys
from tensorboard import default, program
import logging


def parse_file_arg(arg):
    if arg == "fmnist":
        return arg
    else:
        return Path(arg)


class TensorBoardProcess(torch.multiprocessing.Process):

    def __init__(self, output, port=6006):
        super(TensorBoardProcess, self).__init__()
        self.output = output
        self.port = port

    def run(self):
        tb = program.TensorBoard(
            default.get_plugins() + default.get_dynamic_plugins(), 
            program.get_default_assets_zip_provider())
        tb.configure(logdir = str(self.output / "tb"), port = self.port)
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.CRITICAL)
        tb.main()


def main():
    """
    Main entrypoint for this DL suite.
    """

    parent_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(prog="Deep Clustering")
    subparsers = parser.add_subparsers()
    
    group_meta = parent_parser.add_argument_group(title="meta", description="Arguments related to running the program")
    group_meta.add_argument("--cuda", "-u", action="store_true", help="Use cuda")
    group_meta.add_argument("--batch-report-frequency", default=50, type=int)
    group_meta.add_argument("--workers", "-w", default=0, type=int)
    group_meta.add_argument("--launch-tb", "-tb", default=False, action='store_true')

    group_data = parent_parser.add_argument_group(title="data", description="Arguments related to data input.")
    group_data.add_argument("--root", "-r", help="Directory prepended to any path input. (Can be path to dir structure shared accross environments.)", type=parse_file_arg)

    data_group = group_data.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", "-d", help="File containing input images, or 'fmnist'.", default=None, type=str)
    data_group.add_argument("--glob", "-g", help="Glob expression to select filed containing input images.", default=None, type=str)

    group_data.add_argument("--output", "-o", help="Directory for storing output.", default="tmp", type=Path)
    group_data.add_argument("--rm", help="Remove output directory if exists.", action="store_true", default=False)
    group_data.add_argument("--channels", "-c", nargs="*", default=[], type=int, help="Channel indices to be used (only if data is LMDB). Default is to use all.")
    group_data.add_argument("--raw-image", "-ri", action="store_true", default=False, help="Whether to use non-masked image (only if data is LMDB).")
    
    subparser_augtest = subparsers.add_parser(name="augtest", parents=[parent_parser])
    subparser_augtest.set_defaults(func=augtest.main)
    
    subparser_lmdbtest = subparsers.add_parser(name="lmdbtest", parents=[parent_parser])
    subparser_lmdbtest.set_defaults(func=lmdbtest.main)

    parser_model = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    
    group_model_fit = parser_model.add_argument_group(title="fit", description="Arguments related to model fitting")
    group_model_fit.add_argument("--epochs", "-e", help="Number of epochs", default=100, type=int)
    group_model_fit.add_argument("--batch-size", "-bs", type=int, help="Batch size.", default=256)
    group_model_fit.add_argument("--embedding-size", "-es", type=int, help="Embedding size.", default=10)

    subparser_dae = subparsers.add_parser(name="DAE", parents=[parser_model])
    subparser_dae.set_defaults(func=fit_dae.main)

    parser_cae = argparse.ArgumentParser(parents=[parser_model], add_help=False)
    group_cae = parser_cae.add_argument_group(title="CAE specific")
    group_cae.add_argument("--dropout", "-p", default=0.0, type=float)

    subparser_cae = subparsers.add_parser(name="CAE", parents=[parser_cae])
    subparser_cae.set_defaults(func=fit_cae.main)
    
    subparser_mc_cae = subparsers.add_parser(name="MC_CAE", parents=[parser_cae])
    subparser_mc_cae.add_argument("--n-stochastic", "-n", default=10, type=int)
    subparser_mc_cae.set_defaults(func=mc_dropout_cae.main)
        
    subparser_dynAE = subparsers.add_parser(name="dynAE", parents=[parser_model])
    group_dynAE = subparser_dynAE.add_argument_group(title="dynAE specific", description="Arguments related to model fitting with fixed clusters.")
    group_dynAE.add_argument("--clusters", "-N", help="Number of clusters", required=True, type=int)
    group_dynAE.add_argument("--pretrained-model", "-p", required=True, type=parse_file_arg)
    group_dynAE.add_argument("--tolerance", "-t", default=0.0001, type=float)
    subparser_dynAE.set_defaults(func=dyn_ae_clustering.main)
    
    args = parser.parse_args()

    from torch.multiprocessing import set_start_method
    try:
        import os
        method = os.environ.get("DEBUG")
        if os.environ.get("OS") != "Windows_NT" and method is None:
            set_start_method('fork', True)
        else:
            set_start_method('spawn', True)
    except RuntimeError:
        pass

    # specify argument dependencies
    if isinstance(args.data, Path) and args.data.suffix in [".h5", ".hdf5", ".lmdb"]:
        if not hasattr(args, "channels"):
            raise ValueError("Channels is required.")
    if isinstance(args.data, str):
        vars(args)["channels"] = [0]

    # process file arguments

    ## data
    if args.root is not None and args.root.exists():
        # prepend root
        if "pretrained_model" in args and args.pretrained_model is not None:
            args.pretrained_model = args.root / args.pretrained_model
            if not args.pretrained_model.exists():
                raise ValueError("Pretrained model cannot be found.")

        if args.data is not None:
            data = [args.root / Path(args.data)]
        else:
            data = [f for f in Path(args.root).glob(args.glob)]
    elif args.root is not None and not args.root.exists():
        raise ValueError(args.root, "Cannot find root directory.")
    else:
        if args.data is not None:
            data = [Path(args.data)]
        else:
            data = [f for f in Path(".").glob(args.glob)]

    if len(data) == 0:
        raise ValueError("Input files(s) not found.")
    for f in data:
        if not f.exists():
            raise ValueError("Data file does not exists.")

    vars(args)["data"] = [str(d) for d in data]

    # process outputdir arg
    if args.output.exists():
        if args.output == Path("tmp") or args.rm:
            shutil.rmtree(args.output)

    args.output.mkdir()

    # launch tb
    if args.launch_tb:
        tb_process = TensorBoardProcess(args.output)
        tb_process.daemon = True
        tb_process.start()

    torch.manual_seed(42)
    numpy.random.seed(42)

    cleaned_args = {}
    for k, v in vars(args).items():
        if k != "func":
            cleaned_args[k] = v if not isinstance(v, Path) else str(v)
    with open(os.path.join(args.output, "args.json"), "w") as j:
        json.dump(cleaned_args, j)

    args.func(args)


if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
   
    main()
