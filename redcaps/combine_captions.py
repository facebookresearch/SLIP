import argparse
import json
import os


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='', type=str, help='path to redcaps annotations directory')
    parser.add_argument('--output', default='', type=str, help='output annotations file path')
    return parser


def main(args):
    annos = []
    for fname in os.listdir(args.input):
        if fname.endswith('json'):
            with open(os.path.join(args.input, fname)) as f:
                a = json.load(f)
                annos.extend(a['annotations'])

    with open(args.output, 'w') as f:
        json.dump(annos, f)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)