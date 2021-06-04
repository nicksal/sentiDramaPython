import pandas as pd

class XLocIndexer:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, key):
        row, col = key
        return self.frame.iloc[row][col]


pd.core.indexing.IndexingMixin.xloc = property(lambda frame: XLocIndexer(frame))