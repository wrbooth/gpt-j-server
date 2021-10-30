import curses
import argparse
import sys

class Window:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols


def main(stdscr):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        buffer = f.readlines()

    window = Window(curses.LINES - 1, curses.COLS - 1)

    while True:
        stdscr.erase()
        for row, line in enumerate(buffer[:window.n_rows]):
            stdscr.addstr(row, 0, line[:window.n_cols])

        k = stdscr.getkey()
        if k == "q":
            sys.exit(0)


if __name__ == "__main__":
    curses.wrapper(main)
