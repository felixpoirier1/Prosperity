from traders.test1_gio import Trader
from datamodel import *
from utils.DataEngine import DataEngine
import pandas as pd

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

    def process_orders(self, prod_state, order_type):
        """
        Processes bid or ask orders from a product state and updates the orders dictionary.
        """
        price_key_prefix = f'{order_type}_price_'
        volume_key_prefix = f'{order_type}_volume_'
        orders_dict = {}

        for i in range(1, 4):
            price_key = f'{price_key_prefix}{i}'
            volume_key = f'{volume_key_prefix}{i}'
            try:
                orders_dict[int(prod_state[price_key])] = int(prod_state[volume_key])
            except ValueError:
                continue

        return orders_dict

    def run(self):
        timestamp = next(self.timestamp)
        curr_state = self.data.loc[self.data.timestamp == timestamp]
        products = curr_state['product']
        listings = {}
        order_depths = {}
        for product in products:
            prod_state = curr_state[curr_state['product'] == product]
            listings[product] = Listing(product, product, product)
            depths = OrderDepth()
            depths.buy_orders = self.process_orders(prod_state, 'bid')
            depths.sell_orders = self.process_orders(prod_state, 'ask')

            order_depths[product] = depths

        trading_state = TradingState(traderData="", 
                                     timestamp=timestamp,
                                     listings=listings,
                                     order_depths=order_depths,
                                     own_trades=None,
                                     market_trades=None,
                                     position=None,
                                     observations=None)
        self.trader.run(trading_state)
        print(timestamp)

engine = TradingEngine(Trader())
while True:
    engine.run()