import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
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

class Trader:
    def __init__(self) -> None:
        self.position = copy.deepcopy({'STARFRUIT':0, 'AMETHYSTS':0, 'ORCHIDS':0})

        self.cpnl = defaultdict(lambda : 0)
        self.sf_cache = []
        self.POSITION_LIMIT = {'STARFRUIT':20, 'AMETHYSTS':20, 'ORCHIDS':100}
        self.sf_params = [0.08442609, 0.18264657, 0.7329293]

        # orchids
        self.sun_list = []
        self.hum_list = []
        self.sun_len = 3
        self.hum_len = 3
        self.orch_ask = 0

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
                orders.append(Order(product, price, order_for))

        return orders, cpos

    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []
        pos_lim = self.POSITION_LIMIT[product]

        order_s_liq, bcpos = self.liquity_taking(order_depth.sell_orders, acc_bid, True, product, operator.lt)
        order_b_liq, acpos = self.liquity_taking(order_depth.buy_orders, acc_ask, False, product, operator.gt)

        # Market making prices
        if 9998 in order_depth.buy_orders:
            price_bid = 9998
        elif 9996 in order_depth.buy_orders:
            price_bid = 9997
        else:
            price_bid = 9996

        if 10_002 in order_depth.sell_orders:
            price_ask = 10_002
        elif 10_004 in order_depth.sell_orders:
            price_ask = 10_003
        else:
            price_ask = 10_004

        orders += order_s_liq

        # if bcpos <= -15:
        #    price_bid += 1

        #if cpos == -pos_lim:
        #    orders.append(Order(product, 10_002, 1))
        #    cpos += 1

        # Market making bid orders
        if bcpos < pos_lim:
            orders.append(Order(product, price_bid, pos_lim-bcpos))
        
        orders += order_b_liq

        #if acpos >= 15:
        #    price_ask -= 1

        #if cpos == pos_lim:
        #    orders.append(Order(product, 9998, -1))
        #    cpos -= 1

        # Market making ask orders
        if acpos > -pos_lim:
            orders.append(Order(product, price_ask, -pos_lim-acpos))

        return orders

    def compute_starfruit_orders(self, product, order_depth, next_mid):
        orders: list[Order] = []
        lim = self.POSITION_LIMIT[product]

        best_sell_pr, best_buy_pr = self.get_deepest_prices(order_depth)

        if len(self.sf_cache) == len(self.sf_params):
            bid_pr = min(best_buy_pr+1, next_mid-1)
            sell_pr = max(best_sell_pr-1, next_mid+1)
        else:
            bid_pr = best_buy_pr+1
            sell_pr = best_sell_pr-1

        order_s_liq, bcpos = self.liquity_taking(order_depth.sell_orders, next_mid-1, True, product, operator.le)
        orders += order_s_liq

        if bcpos < lim:
            orders.append(Order(product, bid_pr, lim - bcpos))

        if not next_mid:
            next_mid = 1E8
        
        order_b_liq, acpos = self.liquity_taking(order_depth.buy_orders, next_mid+1, False, product, operator.ge)
        orders += order_b_liq

        if acpos > -lim:
            orders.append(Order(product, sell_pr, -lim-acpos))

        return orders

    def find_arbitrage(self, product, order_depth, observation):
        orders: list[Order] = []
        conversions = 0
        cpos = self.position[product]

        ap = observation.askPrice
        it = observation.importTariff
        tf = observation.transportFees

        for bid in order_depth.buy_orders.items():
            if bid[0] - ap - tf - it > 0:
                order_amt = min(bid[1], -cpos)
                if order_amt > 0:
                    orders.append(Order(product, bid[0], -order_amt))
                    cpos += order_amt
                    conversions += order_amt

        return orders, conversions

    def update_orchid_lists(self, observation):
        self.sun_list.append(observation.sunlight)
        if len(self.sun_list) > self.sun_len:
            self.sun_list.pop(0)
        
        self.hum_list.append(observation.humidity)
        if len(self.hum_list) > self.hum_len:
            self.hum_list.pop(0)
    
    def computer_orchids_orders(self, product, order_depth, observation):
        orders: list[Order] = []
        pos_lim = self.POSITION_LIMIT[product]
        apos = self.position[product]
        bpos = self.position[product]
        convsersions = 0
        ap = observation.askPrice
        it = observation.importTariff
        tf = observation.transportFees

        bottom_book, top_buy = sorted(order_depth.buy_orders.items())[0][0], sorted(order_depth.buy_orders.items())[-1][0]
        top_ask_vol, top_buy_vol = sorted(order_depth.sell_orders.items())[0][1], sorted(order_depth.buy_orders.items())[-1][1]

        arb_val = top_buy - ap - tf - it

        self.update_orchid_lists(observation)

        if len(self.sun_list) != self.sun_len:
            return [], 0

        # arb_orders, arb_conversions = self.find_arbitrage(product, order_depth, observation)

        # orders += arb_orders
        # convsersions += arb_conversions
        # bpos += arb_conversions
        # apos -= arb_conversions

        '''
        deriv_sun_1, deriv_sun_2 = self.sun_list[1]-self.sun_list[0], self.sun_list[2]-self.sun_list[1]
        deriv_hum_1, deriv_hum_2 = self.hum_list[1]-self.hum_list[0], self.hum_list[2]-self.hum_list[1]
        
        if np.sign(deriv_sun_2) != np.sign(deriv_sun_1):
            self.sun_reversal = True
            if np.sign(deriv_hum_2) - np.sign(deriv_hum_1) > 0:
                signal = 'buy'
            else:
                signal = 'sell'

        logger.print(f'Derivs: {deriv_hum_1}, {deriv_hum_2}')
        if np.sign(deriv_hum_2) != np.sign(deriv_hum_1):
            if np.sign(deriv_hum_2) - np.sign(deriv_hum_1) > 0:
                signal = 'buy'
            else:
                signal = 'sell'

        if signal == 'buy':
            sig_buy_vol = min(top_ask_vol, pos_lim-bpos)
            orders.append(Order(product, top_ask, sig_buy_vol))
            bpos += sig_buy_vol

        if signal == 'sell':
            sig_ask_vol = max(-top_buy_vol, -pos_lim-apos)
            orders.append(Order(product, top_buy, sig_ask_vol))
            apos -= sig_ask_vol
        '''
        sell_pr_1 = top_buy+2
        sell_pr_2 = top_buy+3

        if apos < 0:
            if arb_val-1 > 0:
                convsersions -= bpos
                apos += convsersions
                bpos -= convsersions
                sell_pr_1 = top_buy
                sell_pr_2 = top_buy+1
            elif arb_val > 0:
                convsersions -= bpos
                apos += convsersions
                bpos -= convsersions
                sell_pr_1 = top_buy+1
                sell_pr_2 = top_buy+2
            elif arb_val+1 > 0:
                convsersions -= bpos
                apos += convsersions
                bpos -= convsersions
                sell_pr_1 = top_buy+2
                sell_pr_2 = top_buy+3
            else:
                orders.append(Order(product, top_buy+1, -bpos))

        sell_vol_1 = int((-pos_lim-apos)*0.5)
        sell_vol_2 = -pos_lim-apos-sell_vol_1

        if apos > -pos_lim:
            orders.append(Order(product, sell_pr_1, sell_vol_1))
            orders.append(Order(product, sell_pr_2, sell_vol_2))

        return orders, convsersions

    def deserializeJson(self, json_string):
        if json_string == "":
            logger.print("Empty trader data")
            return
        state_dict = jsonpickle.decode(json_string)
        for key, value in state_dict.items():
            setattr(self, key, value)
    
    def serializeJson(self):
        return jsonpickle.encode(self.__dict__)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        try:
            self.deserializeJson(state.traderData)
        except:
            logger.print("Error in deserializing trader data")

        result = {}

        for key, val in state.position.items():
            self.position[key] = val

        # To be changed later
        trader_data = state.traderData
        conversion_observation = state.observations.conversionObservations
        orchid_observation = conversion_observation['ORCHIDS']

        if len(self.sf_cache) == len(self.sf_params):
            self.sf_cache.pop(0)

        bs_starfruit, bb_starfruit = self.get_deepest_prices(state.order_depths['STARFRUIT'])

        self.sf_cache.append((bs_starfruit+bb_starfruit)/2)
    
        next_price = 0

        if len(self.sf_cache) == len(self.sf_params):
            next_price = int((np.array(self.sf_cache) * np.array(self.sf_params)).sum())

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            if product == 'AMETHYSTS':
                result[product] = self.compute_orders_amethysts(product, order_depth, 10_000, 10_000)
            elif product == 'STARFRUIT':
                result[product] = self.compute_starfruit_orders(product, order_depth, next_price)
            elif product == 'ORCHIDS':
                orchid_orders, conversions = self.computer_orchids_orders(product, order_depth, orchid_observation)
                result[product] = orchid_orders

        trader_data = self.serializeJson()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data