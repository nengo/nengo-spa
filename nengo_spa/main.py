"""Command line interface"""

import argparse

from nengo_spa.resources import extract_resource


def main(argv):
    parser = argparse.ArgumentParser('nengo_spa')
    subparsers = parser.add_subparsers()

    parser_extract_examples = subparsers.add_parser(
        'extract-examples',
        description="Extract nengo_spa examples to given destination.",
        help="Extract examples")
    parser_extract_examples.add_argument('dest', type=str, help="destination")
    parser_extract_examples.set_defaults(
        func=lambda args: extract_resource('examples', args.dest))

    args = parser.parse_args(argv[1:])
    args.func(args)
