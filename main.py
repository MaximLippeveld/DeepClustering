import argparse
from func import augtest, fit_dae
import os

def parse_file_arg(arg):
    if os.path.exists(arg):
        return arg

    argparse.ArgumentError(arg, "The file does not exist.")

def main():
    """
    Main entrypoint for this DL suite.
    """

    parent_parser = argparse.ArgumentParser(add_help=False)

    group_data = parent_parser.add_argument_group(title="data", description="Arguments related to data input.")
    group_data.add_argument("--root", "-r", help="Directory prepended to any path input. (Can be path to dir structure shared accross environments.)")
    group_data.add_argument("--data", "-d", help="HDF5-file containing input images.", required=True, type=parse_file_arg)
    group_data.add_argument("--channels", "-c", nargs="*", type=int, help="Channel numbers to be used.", required=True)
    group_data.add_argument("--batch-size", "-b", type=int, help="Batch size.", default=256)

    parser = argparse.ArgumentParser(prog="uncertainty-in-dl")
    subparsers = parser.add_subparsers()

    subparser_augtest = subparsers.add_parser(name="augtest", parents=[parent_parser])
    subparser_augtest.set_defaults(func=augtest.main)

    parser_model = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    
    group_model_fit = parser_model.add_argument_group(title="fit", description="Arguments related to model fitting")
    group_model_fit.add_argument("--epochs", "-e", help="Number of epochs", default=100, type=int)

    subparser_dae = subparsers.add_parser(name="DAE", parents=[parser_model])
    subparser_dae.set_defaults(func=fit_dae.main)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()