"""Entry point for the Dalcedo CLI."""

import sys


def main() -> int:
    """Main entry point for the application."""
    from dalcedo.app import DalcedoApp

    app = DalcedoApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
