# src/cli.py
import argparse
from separation.main import RealtimeSeparatorEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="vocals")
    args = parser.parse_args()

    engine = RealtimeSeparatorEngine(stem=args.stem)
    engine.run()

if __name__ == "__main__":
    main()