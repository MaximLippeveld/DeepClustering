import argparse
from func import augtest, fit_dae, fit_cae
import os
import importlib
from pathlib import Path

def parse_file_arg(arg):
    if arg == "fmnist":
        return arg
    else:
        return Path(arg)

def main():
    """
    Main entrypoint for this DL suite.
    """

    parent_parser = argparse.ArgumentParser(add_help=False)

    group_meta = parent_parser.add_argument_group(title="meta", description="Arguments related to running the program")
    group_meta.add_argument("--cuda", "-u", action="store_true", help="Use cuda")

    group_data = parent_parser.add_argument_group(title="data", description="Arguments related to data input.")
    group_data.add_argument("--root", "-r", help="Directory prepended to any path input. (Can be path to dir structure shared accross environments.)", type=parse_file_arg)
    group_data.add_argument("--data", "-d", help="File containing input images or 'fmnist'.", required=True, type=parse_file_arg)
    group_data.add_argument("--channels", "-c", nargs="*", type=int, help="Channel numbers to be used (only if data is HDF5).")
    group_data.add_argument("--batch-size", "-b", type=int, help="Batch size.", default=256)

    parser = argparse.ArgumentParser(prog="Deep Clustering")
    subparsers = parser.add_subparsers()

    subparser_augtest = subparsers.add_parser(name="augtest", parents=[parent_parser])
    subparser_augtest.set_defaults(func=augtest.main)

    parser_model = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    
    group_model_fit = parser_model.add_argument_group(title="fit", description="Arguments related to model fitting")
    group_model_fit.add_argument("--epochs", "-e", help="Number of epochs", default=100, type=int)

    subparser_dae = subparsers.add_parser(name="DAE", parents=[parser_model])
    subparser_dae.set_defaults(func=fit_dae.main)
    
    subparser_cae = subparsers.add_parser(name="CAE", parents=[parser_model])
    subparser_cae.set_defaults(func=fit_cae.main)
    
    args = parser.parse_args()

    # cuda specific setup
    if args.cuda:
        from torch.multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

    # specify argument dependencies
    if ".hdf5" in args.data:
        if not hasattr(args, "channels"):
            raise argparse.ArgumentError("Channels is required.")

    # process file arguments
    if type(args.data) is Path:
        if not args.data.exists() and args.root.exists():
            if (args.data / args.root).exists():
                args.data /= args.root
            else:
                raise argparse.ArgumentError("Concatentation of root and data does not exist.")
        else:
            raise argparse.ArgumentError("Data arg does not exist and root does not exist.")

    args.func(args)


if __name__ == "__main__":
    main()