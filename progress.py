import sys
from math import floor


class ProgressBar:
    def __init__(self, total, length=15, auto_linechange=True):
        self.total = total
        self.length = length
        self.current = 0
        self.auto_linechange = auto_linechange

    def add(self, value):
        self.current += value
        self.refresh()

    def refresh(self):
        print("\r", end="")
        cnt = floor(self.current / self.total * self.length)
        print(f"[{'=' * cnt}{' ' * (self.length - cnt)}] {self.current}/{self.total}", end="")
        if self.current >= self.total and self.auto_linechange:
            print()
        sys.stdout.flush()
