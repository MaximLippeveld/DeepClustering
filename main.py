import argparse
from func import augtest
import os

def parse_file_arg(arg):
    if os.path.exists(arg):
        return arg

    argparse.ArgumentError(arg, "The file does not exist.")

def main():
    """
    Main entrypoint for this DL suite.
    """

    parser = argparse.ArgumentParser(prog="uncertainty-in-dl")
    subparsers = parser.add_subparsers()

    group_data = parser.add_argument_group(title="data", description="Arguments related to data input.")
    group_data.add_argument("--root", "-r", help="Directory prepended to any path input. (Can be path to dir structure shared accross environments.)")
    group_data.add_argument("--data", "-d", help="HDF5-file containing input images.", required=True, type=parse_file_arg)
    group_data.add_argument("--channels", "-c", nargs="*", type=int, help="Channel numbers to be used.", required=True)

    parser_data = subparsers.add_parser(name="augtest")
    parser_data.set_defaults(func=augtest.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()