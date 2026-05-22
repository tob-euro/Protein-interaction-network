import argparse

import yaml

from src.training.runner import run_single


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='config/config.yaml', help='YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg['data']['random_state']
    metrics = run_single(cfg, seed, args.config)

    print(f"\n--- Training complete ---")
    print(f"Test AUC: {metrics['test_auc']:.4f}  |  AP: {metrics['test_ap']:.4f}")
    print(f"Saved to: {metrics['save_dir']}\n")


if __name__ == "__main__":
    main()
