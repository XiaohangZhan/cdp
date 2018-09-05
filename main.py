import yaml
import argparse
from knn import create_knn
from cdp import cdp

def main():
    parser = argparse.ArgumentParser(description="CDP")
    parser.add_argument('--config', default='', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    for k,v in config.items():
        setattr(args, k, v)

    assert isinstance(args.committee, list), "committee should be a list of strings"
    create_knn(args)
    cdp(args)

if __name__ == "__main__":
    main()
