import argparse
import yaml
from src.data_scripts.split_cache import SplitCache
from src.training.runner import run_single


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',    default='config/config.yaml', help='YAML config file')
    parser.add_argument('--cache',     action='store_true',
                        help='Enable disk cache for data splits (off by default)')
    parser.add_argument('--cache_dir', default='.cache/data_splits',
                        help='Directory for cached split files')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_cache = SplitCache(args.cache_dir) if args.cache else None
    seed = cfg['data']['random_state']
    metrics = run_single(cfg, seed, args.config, split_cache=split_cache)

    print(f"\n--- Training complete ---")
    print(f"Test AUC: {metrics['test_auc']:.4f}  |  AP: {metrics['test_ap']:.4f}")
    print(f"Saved to: {metrics['save_dir']}\n")


if __name__ == "__main__":
    main()
