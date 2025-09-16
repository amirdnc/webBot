import argparse

DEFAULT_GOAL = "what is the weather in tokyo?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Goal-oriented web agent")
    parser.add_argument(
        "--goal",
        default=DEFAULT_GOAL,
        help="Goal the agent should accomplish.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging of planned and executed tasks.",
    )
    parser.add_argument(
        "--log-file",
        help="Path to write verbose log output.",
    )
    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the agent."""
    parser = build_parser()
    return parser.parse_args(args)
