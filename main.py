import argparse
from func import augtest


def main():
    """
    Main entrypoint for this DL suite.
    """

    parser = argparse.ArgumentParser(prog="uncertainty-in-dl")
    subparsers = parser.add_subparsers()

    group_data = parser.add_argument_group(title="data", description="Arguments related to data input.")
    group_data.add_argument("--data", "-d", type=argparse.FileType(mode="rb"), help="HDF5-file containing input images.", required=True)
    group_data.add_argument("--channels", "-c", nargs="*", type=int, help="Channel numbers to be used.", required=True)

    parser_data = subparsers.add_parser(name="augtest")
    parser_data.set_defaults(func=augtest.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()