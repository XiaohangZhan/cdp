import yaml
import os
import argparse
import logging
from source.knn import create_knn
from source.cdp import cdp

def main():
    parser = argparse.ArgumentParser(description="CDP")
    parser.add_argument('--config', default='', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    for k,v in config.items():
        setattr(args, k, v)

    log_path = "{}/output/log.txt".format(os.path.dirname(args.config))
    if not os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logging.basicConfig(filename=log_path)

    assert isinstance(args.committee, list), "committee should be a list of strings"

    if args.strategy == "mediator":
        create_knn(args, args.mediator['train_data_name'])
    create_knn(args, args.data_name)
    cdp(args)

if __name__ == "__main__":
    main()
