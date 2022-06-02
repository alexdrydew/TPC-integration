import yaml
import argparse


def main(args):
    with open(args.config, "r") as f:
        parsed = yaml.safe_load(f)
        parsed["security"] = [{opt: []} for opt in args.auth_options]
    with open(args.config, "w") as f:
        yaml.dump(parsed, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--auth_options", nargs="+", required=True)

    main(parser.parse_args())
