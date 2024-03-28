import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import collections
from collections import defaultdict
import numpy as np
import copy
import operator


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
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
    sf_cache = []
    POSITION_LIMIT = {'STARFRUIT':20, 'AMETHYSTS':20} 
    sf_params = [15.227814, 0.453990, 0.542945] 

    def get_deepest_prices(self, order_depth):
        best_sell_pr = sorted(order_depth.sell_orders.items())[-1][0]
        best_buy_pr = sorted(order_depth.buy_orders.items())[0][0]

        return best_sell_pr, best_buy_pr

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

    def compute_starfruit_orders(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []
        lim = self.POSITION_LIMIT[product]

        best_sell_pr, best_buy_pr = self.get_deepest_prices(order_depth)

        bid_pr = min(best_buy_pr+1, acc_bid)
        sell_pr = max(best_sell_pr-1, acc_ask)

        order_s_liq, cpos = self.liquity_taking(order_depth.sell_orders, acc_bid, True, product, operator.le)
        orders += order_s_liq

        if cpos < lim:
            orders.append(Order(product, bid_pr, lim - cpos))
        
        order_b_liq, cpos = self.liquity_taking(order_depth.buy_orders, acc_ask, False, product, operator.ge)
        orders += order_b_liq

        if cpos > -lim:
            orders.append(Order(product, sell_pr, -lim-cpos))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}

        for key, val in state.position.items():
            self.position[key] = val

        # To be changed later
        conversions = 0
        trader_data = ""

        if len(self.sf_cache) == (len(self.sf_params) - 1):
            self.sf_cache.pop(0)

        bs_starfruit, bb_starfruit = self.get_deepest_prices(state.order_depths['STARFRUIT'])

        self.sf_cache.append((bs_starfruit+bb_starfruit)/2)

        INF = 1e9
    
        starfruit_lb = -INF
        starfruit_ub = INF

        if len(self.sf_cache) == (len(self.sf_params) - 1):
            next_price = int((np.array(self.sf_cache) * np.array(self.sf_params[1:])).sum() + self.sf_params[0])

            starfruit_lb = next_price-1
            starfruit_ub = next_price+1

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            if product == 'AMETHYSTS':
                result[product] = self.compute_orders_amethysts(product, order_depth, 10_000, 10_000)
            elif product == 'STARFRUIT':
                result[product] = self.compute_starfruit_orders(product, order_depth, starfruit_lb, starfruit_ub)
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data