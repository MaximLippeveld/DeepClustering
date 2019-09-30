import argparse
from func import augtest, fit_dae, fit_cae, fit_dyn_ae
import os
import importlib
from pathlib import Path
import shutil

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
    parser = argparse.ArgumentParser(prog="Deep Clustering")
    subparsers = parser.add_subparsers()
    
    subparser_augtest = subparsers.add_parser(name="augtest", parents=[parent_parser])
    subparser_augtest.set_defaults(func=augtest.main)

    group_meta = parent_parser.add_argument_group(title="meta", description="Arguments related to running the program")
    group_meta.add_argument("--cuda", "-u", action="store_true", help="Use cuda")

    group_data = parent_parser.add_argument_group(title="data", description="Arguments related to data input.")
    group_data.add_argument("--root", "-r", help="Directory prepended to any path input. (Can be path to dir structure shared accross environments.)", type=parse_file_arg)
    group_data.add_argument("--data", "-d", help="File containing input images or 'fmnist'.", required=True, type=parse_file_arg)
    group_data.add_argument("--output", "-o", help="Directory for storing output.", default="tmp", type=Path)
    group_data.add_argument("--rm", help="Remove output directory if exists.", action="store_true", default=False)
    group_data.add_argument("--channels", "-c", nargs="*", type=int, help="Channel numbers to be used (only if data is HDF5).")

    parser_model = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    
    group_model_fit = parser_model.add_argument_group(title="fit", description="Arguments related to model fitting")
    group_model_fit.add_argument("--epochs", "-e", help="Number of epochs", default=100, type=int)
    group_model_fit.add_argument("--batch-size", "-b", type=int, help="Batch size.", default=256)
    group_model_fit.add_argument("--embedding-size", "-s", type=int, help="Embedding size.", default=10)
    group_model_fit.add_argument("--pretrained-model", "-p", type=parse_file_arg)
    group_model_fit.add_argument("--tolerance", "-t", default=0.0001, type=float)

    subparser_dae = subparsers.add_parser(name="DAE", parents=[parser_model])
    subparser_dae.set_defaults(func=fit_dae.main)
    
    subparser_cae = subparsers.add_parser(name="CAE", parents=[parser_model])
    subparser_cae.set_defaults(func=fit_cae.main)
    
    parser_fixed = argparse.ArgumentParser(parents=[parser_model], add_help=False)

    group_fixed_fit = parser_fixed.add_argument_group(title="fit", description="Arguments related to model fitting with fixed clusters.")
    group_fixed_fit.add_argument("--clusters", "-N", help="Number of clusters", required=True, type=int)
    
    subparser_cae = subparsers.add_parser(name="dynAE", parents=[parser_fixed])
    subparser_cae.set_defaults(func=fit_dyn_ae.main)
    
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
            raise ValueError("Channels is required.")

    # process file arguments
    for p in [args.data, args.pretrained_model]:
        if isinstance(p, Path):
            if not p.exists():
                if args.root is not None and args.root.exists():
                    if (p / args.root).exists():
                        p /= args.root
                    else:
                        raise ValueError("Concatenation of root and %s does not exist." % p)
                else:
                    raise ValueError("%s does not exist and root does not exist." % p)
    
    # process outputdir arg
    if args.output.exists():
        if args.output == Path("tmp") or args.rm:
            shutil.rmtree(args.output)

    args.output.mkdir() 

    args.func(args)


if __name__ == "__main__":
    main()