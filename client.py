import curses
import argparse


def main(stdscr):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        buffer = f.readlines()

    while True:
        stdscr.erase()
        for row, line in enumerate(buffer):
            stdscr.addstr(row, 0, line)

        k = stdscr.getkey()
        if k == "q":
            sys.exit(0)

if __name__ == "__main__":
    curses.wrapper(main)
