import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-i', '--image',
        dest='image',
        metavar='I',
        default='None',
        help='Path of image')
    args = argparser.parse_args()
    return args
