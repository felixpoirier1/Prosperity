'''
Keep in mind that implementing the same code can return two different PnLs. Check with mods. Pm jacek.
'''


import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import collections
import os
from collections import defaultdict
import numpy as np
import copy
import operator

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


empty_dict = {'STARFRUIT':0, 'AMETHYSTS':0} 

def def_value():
    return copy.deepcopy(empty_dict)


class Trader:
    position = copy.deepcopy(empty_dict)
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda : 0)
    sf_bid_cache = []
    sf_ask_cache = []
    sf_mid_cache = []
    POSITION_LIMIT = {'STARFRUIT':20, 'AMETHYSTS':20} 
    sf_bid_params = [10.218076238395952, 0.12510656, 0.10468357, 0.20735852, 0.56092445]
    sf_ask_params = [-39.70531615296659, 0.07975563, 0.12160025, 0.24590879, 0.56051938]
    sf_mid_params = [-16.368438120524843, 0.07294425, 0.13279522, 0.19364201, 0.60387595]

    def get_deepest_prices(self, order_depth):
        best_sell_pr = sorted(order_depth.sell_orders.items())[-1][0]
        best_buy_pr = sorted(order_depth.buy_orders.items())[0][0]

        return best_buy_pr+1, best_sell_pr-1

    def get_price_condition(self, is_sell, price, acc_price, product, comp_op):
        first_val = comp_op(price, acc_price)
        if is_sell:
            if product == 'STARFRUIT':
                acc_price += 1
            return (first_val or ((self.position[product]<0) and (price == acc_price))) 
        else:
           if product == 'STARFRUIT':
                price += 1
           return (first_val or ((self.position[product]>0) and (price == acc_price))) 

    def liquity_taking(self, orders_dict, acc_price, is_sell, product, comp_op):
        cpos = self.position[product]
        pos_limit = self.POSITION_LIMIT[product]
        orders = []

        for price, vol in orders_dict.items():
            price_condition = self.get_price_condition(is_sell, price, acc_price, product, comp_op)
            position_condition = cpos < pos_limit if is_sell else cpos > -pos_limit
            if price_condition and position_condition:
                order_for = min(-vol, pos_limit-cpos) if is_sell else max(-vol, -pos_limit-cpos)
                cpos += order_for
                assert(order_for >=0 if is_sell else order_for <= 0)
                orders.append(Order(product, price, order_for))

        return orders, cpos

    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []
        pos_lim = self.POSITION_LIMIT[product]

        order_s_liq, cpos = self.liquity_taking(order_depth.sell_orders, acc_bid, True, product, operator.lt)
        orders += order_s_liq

        # Market making prices
        price_bid = 9997 if 9996 in order_depth.buy_orders else 9996
        price_ask = 10_003 if 10_004 in order_depth.sell_orders else 10_004

        # Market making bid orders
        if cpos < pos_lim:
            orders.append(Order(product, price_bid, pos_lim-cpos))
        
        order_b_liq, cpos = self.liquity_taking(order_depth.buy_orders, acc_ask, False, product, operator.gt)
        orders += order_b_liq

        # Market making ask orders
        if cpos > -pos_lim:
            orders.append(Order(product, price_ask, -pos_lim-cpos))

        return orders
    
    def compute_sf_weighted_avg(self, order_depth):
        weighted_ask = 0
        weighted_bid = 0
        ask_vol = 0
        bid_vol = 0

        for order_ask in order_depth.sell_orders.items():
            ask_vol += -order_ask[1]

        for order_bid in order_depth.buy_orders.items():
            bid_vol += order_bid[1]

        for order_ask in order_depth.sell_orders.items():
            weighted_ask += order_ask[0]*-(order_ask[1]/ask_vol)

        for order_bid in order_depth.buy_orders.items():
            weighted_bid += order_bid[0]*(order_bid[1]/bid_vol)

        return weighted_bid, weighted_ask

    def sf_caches(self, order_depth):
        if len(self.sf_bid_cache) == (len(self.sf_bid_params) - 1):
            self.sf_bid_cache.pop(0)
        
        if len(self.sf_ask_cache) == (len(self.sf_ask_params) - 1):
            self.sf_ask_cache.pop(0)
        
        if len(self.sf_mid_cache) == (len(self.sf_mid_params) - 1):
            self.sf_mid_cache.pop(0)

        weighted_bid, weighted_ask = self.compute_sf_weighted_avg(order_depth)
        self.sf_bid_cache.append(weighted_bid)
        self.sf_ask_cache.append(weighted_ask)
        self.sf_mid_cache.append((weighted_bid+weighted_ask)/2)

    def calc_sf_prices(self, order_depth, next_bid, next_ask):
        if not next_bid or not next_ask:
            return self.get_deepest_prices(order_depth)

        bid_pr = sorted(order_depth.buy_orders.items())[-1][0]
        ask_pr = sorted(order_depth.sell_orders.items())[0][0]

        if bid_pr-1 > next_bid:
            try:
                bid_pr = min(next_bid, sorted(order_depth.buy_orders.items())[-2][0]+1)
            except Exception as e:
                bid_pr = next_bid
        else:
            bid_pr += 1
        
        if ask_pr+1 < next_ask:
            try:
                ask_pr = max(next_ask, sorted(order_depth.sell_orders.items())[1][0]-1)
            except Exception as e:
                ask_pr = next_ask
        else:
            ask_pr -= 1

        return bid_pr, ask_pr

    def compute_starfruit_orders(self, product, order_depth, next_bid, next_ask, next_mid):
        orders: list[Order] = []
        lim = self.POSITION_LIMIT[product]

        bid_pr, ask_pr = self.calc_sf_prices(order_depth, next_bid, next_ask)

        order_s_liq, cpos = self.liquity_taking(order_depth.sell_orders, next_mid-1, True, product, operator.le)
        orders += order_s_liq

        if cpos < lim:
            orders.append(Order(product, bid_pr, lim - cpos))
        
        order_b_liq, cpos = self.liquity_taking(order_depth.buy_orders, next_mid+1, False, product, operator.ge)
        orders += order_b_liq

        if cpos > -lim:
            orders.append(Order(product, ask_pr, -lim-cpos))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        if "debug_backtester.log" not in os.listdir():
            with open("debug_backtester.log", "w") as f:
                f.write("")
        else:
            with open("debug_backtester.log", "a") as f:
                f.write(state.toJSON())
                f.write("\n")

        result = {}

        for key, val in state.position.items():
            self.position[key] = val

        # To be changed later
        conversions = 0
        trader_data = ""

        self.sf_caches(state.order_depths['STARFRUIT'])

        next_bid, next_ask, next_mid = (0, 0 ,0)

        if len(self.sf_mid_cache) == (len(self.sf_mid_params) - 1):
            next_bid = int((np.array(self.sf_bid_cache) * np.array(self.sf_bid_params[1:])).sum() + self.sf_bid_params[0])
            next_ask = int((np.array(self.sf_ask_cache) * np.array(self.sf_ask_params[1:])).sum() + self.sf_ask_params[0])
            next_mid = int((np.array(self.sf_mid_cache) * np.array(self.sf_mid_params[1:])).sum() + self.sf_mid_params[0])

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            if product == 'AMETHYSTS':
                result[product] = self.compute_orders_amethysts(product, order_depth, 10_000, 10_000)
            elif product == 'STARFRUIT':
                result[product] = self.compute_starfruit_orders(product, order_depth, next_bid, next_ask, next_mid)
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data