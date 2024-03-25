from trader import Trader
from datamodel import *
from utils.DataEngine import DataEngine

class Timestamp:
    def __init__(self, timestamp=0):
        self.timestamp = timestamp
    
    def __next__(self):
        self.timestamp += 100
        return self.timestamp
    def __iter__(self):
        return self

class TradingEngine:
    def __init__(self, trader):
        self.trader: Trader = trader
        self.data = DataEngine('data2023').data["round_1"]["price_df"]
        self.timestamp = Timestamp()

    def run(self):
        timestamp = next(self.timestamp)
        curr_state = self.data.loc[self.data.timestamp == timestamp]
        print(curr_state)
        #trading_state = TradingState(traderData="", **curr_state)
        #self.trader.run()

engine = TradingEngine(Trader())
while True:
    engine.run()