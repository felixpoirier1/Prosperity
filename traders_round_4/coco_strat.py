import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict, Tuple
from collections import defaultdict
import numpy as np
import copy
import operator
import math


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
        self.position = {'STARFRUIT':0, 'AMETHYSTS':0, 'ORCHIDS':0, 'CHOCOLATE': 0, 'ROSES': 0, "GIFT_BASKET": 0 , 'STRAWBERRIES': 0, "COCONUT": 0, "COCONUT_COUPON": 0}

        self.cpnl = defaultdict(lambda : 0)
        self.sf_cache = []
        self.POSITION_LIMIT = {'STARFRUIT':20, 'AMETHYSTS':20, 'ORCHIDS':100, 'CHOCOLATE': 250, 'ROSES': 60, "GIFT_BASKET": 60 , 'STRAWBERRIES': 350, "COCONUT": 300, "COCONUT_COUPON": 600}
        self.sf_params = [0.08442609, 0.18264657, 0.7329293]

        # orchids
        self.orch_ask = 0
        self.orch_bid = 0
        self.arb = None

        # etf
        self.spread_std = 75
        self.synth_premium = 375
        self.side = None

        # option
        self.coup_spread = 2
        self.coco_spread = 10
        self.count = 0

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

        # Market making bid orders
        if bcpos < pos_lim:
            orders.append(Order(product, price_bid, pos_lim-bcpos))
        
        orders += order_b_liq

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

    def get_orchid_prices(self, added_val, top_ask, top_buy, second_ask):
        mid_price = int((top_buy+top_ask)/2)
        if added_val <= 1.168099856680999:
            self.arb = 'low'
        elif added_val >= 2.168099856680999:
            self.arb = 'high'
        else:
            self.arb = 'mid'

        if top_ask - top_buy > 4:
            # Mean value of import - fees
            if self.arb == 'low':
                ask_pr = mid_price+1
            elif self.arb == 'high':
                ask_pr = mid_price-1
            else:
                ask_pr = mid_price
        else:
            if top_ask-top_buy == 4:
                if self.arb == 'high':
                    ask_pr = top_ask-2
                elif self.arb == 'low':
                    ask_pr = top_ask
                else:
                    ask_pr = top_ask-1
            else:
                if second_ask-top_ask == 1:
                    ask_pr = top_ask-1
                else:
                    if self.arb == 'low':
                        ask_pr = top_ask + 1
                    else:
                        ask_pr = top_ask
            
        return ask_pr
    
    def compute_orchids_orders(self, product, order_depth, observation):
        orders: list[Order] = []
        pos_lim = self.POSITION_LIMIT[product]
        apos = self.position[product]
        bpos = apos
        convsersions = 0

        ap = observation.askPrice
        bp = observation.bidPrice
        imp = observation.importTariff
        exp = observation.exportTariff
        fee = observation.transportFees

        top_ask, top_buy = sorted(order_depth.sell_orders.items())[0][0], sorted(order_depth.buy_orders.items())[-1][0]
        second_ask = sorted(order_depth.sell_orders.items())[0][0]
        ask_pr = self.get_orchid_prices(-imp-fee, top_ask, top_buy, second_ask)

        ask_arb_val = max(ask_pr, self.orch_ask)

        arb_val = ask_arb_val-ap-imp-fee

        if self.arb == 'high' or self.arb == 'mid':
            if apos < 0:
                if arb_val > 0:
                    convsersions -= bpos
                    apos += convsersions
                    bpos -= convsersions
            
            if apos > -pos_lim:
                orders.append(Order(product, ask_pr, -pos_lim-apos))
                self.orch_ask = ask_pr
        
        elif self.arb == 'low':
            cost_buy = ap - fee + imp
            cost_sell = bp - fee - exp

            if cost_buy < top_buy:
                quantity_to_sell = min(self.POSITION_LIMIT['ORCHIDS'] - apos, 50)
                orders.append(Order('ORCHIDS', top_buy, -quantity_to_sell))
                convsersions = quantity_to_sell

            if cost_sell > top_ask:
                quantity_to_buy = min(self.POSITION_LIMIT['ORCHIDS'] - apos, 50)
                orders.append(Order('ORCHIDS', top_ask, quantity_to_buy))
                convsersions = -quantity_to_buy

        return orders, convsersions

    def _compute_synthetic_prices(self, etf_components: Dict[Symbol, OrderDepth]) -> Tuple[int, int]:
        straw_buy = sorted(etf_components["STRAWBERRIES"].buy_orders.items(), reverse=True)[0][0]
        straw_sell = sorted(etf_components["STRAWBERRIES"].sell_orders.items())[0][0]
        choco_buy = sorted(etf_components["CHOCOLATE"].buy_orders.items(), reverse=True)[0][0]
        choco_sell = sorted(etf_components["CHOCOLATE"].sell_orders.items())[0][0]
        rose_buy = sorted(etf_components["ROSES"].buy_orders.items(), reverse=True)[0][0]
        rose_sell = sorted(etf_components["ROSES"].sell_orders.items())[0][0]
        
        synth_base_bid = (6*straw_buy + 4*choco_buy + rose_buy)
        synth_base_ask = (6*straw_sell + 4*choco_sell + rose_sell)

        return int(synth_base_bid)+self.synth_premium, int(synth_base_ask)+self.synth_premium
        
    def _assess_etf_arbitrage(self, etf: OrderDepth, synth_bid: int, synth_ask) -> str:
        etf_ask = sorted(etf.sell_orders.items())[0][0]
        etf_bid = sorted(etf.buy_orders.items(), reverse=True)[0][0]

        etf_mid = (etf_ask + etf_bid)/2
        synth_mid = (synth_ask + synth_bid)/2

        if synth_bid-etf_ask > 1*self.spread_std:
            self.side = "undervalued"
        elif etf_bid-synth_ask > 1*self.spread_std:
            self.side = "overvalued"
        elif (synth_mid-etf_mid < 0*self.spread_std) and self.position['GIFT_BASKET'] > 0:
            self.side = 'rebalance_under'
        elif (etf_mid-synth_mid < 0*self.spread_std) and self.position['GIFT_BASKET'] < 0:
            self.side = 'rebalance_over'
        else:
            self.side = None
        
    def _compute_etf_orders(self, order_depths: Dict[Symbol, OrderDepth]) -> dict[Symbol, list[Order]]:
        orders: Dict[Symbol, list[Order]] = {"GIFT_BASKET": [], 'CHOCOLATE':[], 'STRAWBERRIES':[], 'ROSES':[]}

        gift_sell = sorted(order_depths["GIFT_BASKET"].sell_orders.items())[0][0]
        gift_buy = sorted(order_depths["GIFT_BASKET"].buy_orders.items(), reverse=True)[0][0]
        choco_sell = sorted(order_depths["CHOCOLATE"].sell_orders.items())[0][0]
        choco_buy = sorted(order_depths["CHOCOLATE"].buy_orders.items(), reverse=True)[0][0]
        straw_sell = sorted(order_depths["STRAWBERRIES"].sell_orders.items())[0][0]
        straw_buy = sorted(order_depths["STRAWBERRIES"].buy_orders.items(), reverse=True)[0][0]
        roses_sell = sorted(order_depths["ROSES"].sell_orders.items())[0][0]
        roses_buy = sorted(order_depths["ROSES"].buy_orders.items(), reverse=True)[0][0]

        if self.side == 'undervalued':
            pos_mins = min(
                    (self.POSITION_LIMIT["STRAWBERRIES"]+self.position["STRAWBERRIES"])//6, 
                    (self.POSITION_LIMIT["CHOCOLATE"]+self.position["CHOCOLATE"])//4, 
                    (self.POSITION_LIMIT["ROSES"]+self.position["ROSES"]), 
                    (self.POSITION_LIMIT["GIFT_BASKET"]-self.position["GIFT_BASKET"])
                    )
            
            vol = min(1, pos_mins)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_sell, vol))
            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_buy, -vol*4))
            orders['ROSES'].append(Order('ROSES', roses_buy, -vol))
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_buy, -vol*6))

        elif self.side == 'overvalued':
            pos_mins = min(
                (self.POSITION_LIMIT["STRAWBERRIES"]-self.position["STRAWBERRIES"])//6, 
                (self.POSITION_LIMIT["CHOCOLATE"]-self.position["CHOCOLATE"])//4, 
                (self.POSITION_LIMIT["ROSES"]-self.position["ROSES"]), 
                (self.POSITION_LIMIT["GIFT_BASKET"]+self.position["GIFT_BASKET"])
                )
            vol = min(1, pos_mins)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_buy, -vol))
            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_sell, vol*4))
            orders['ROSES'].append(Order('ROSES', roses_sell, vol))
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_sell, vol*6))
        elif self.side == 'rebalance_under':
            pos_mins = min(
                (-self.position["STRAWBERRIES"])//6, 
                (-self.position["CHOCOLATE"])//4, 
                (-self.position["ROSES"]), 
                (self.position["GIFT_BASKET"])
                )
            vol = min(1, pos_mins)
            if vol != 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_buy, -vol))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_sell, vol*4))
                orders['ROSES'].append(Order('ROSES', roses_sell, vol))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_sell, vol*6))
            else:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_buy, -self.position["GIFT_BASKET"]))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_sell, -self.position["CHOCOLATE"]))
                orders['ROSES'].append(Order('ROSES', roses_sell, -self.position["ROSES"]))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_sell, -self.position["STRAWBERRIES"]))
        elif self.side == 'rebalance_over':
            pos_mins = min(
                    (self.position["STRAWBERRIES"])//6, 
                    (self.position["CHOCOLATE"])//4, 
                    (self.position["ROSES"]), 
                    (-self.position["GIFT_BASKET"])
                    )
            vol = min(1, pos_mins)
            if vol != 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_sell, vol))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_buy, -vol*4))
                orders['ROSES'].append(Order('ROSES', roses_buy, -vol))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_buy, -vol*6))
            else:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_sell, -self.position["GIFT_BASKET"]))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_buy, -self.position["CHOCOLATE"]))
                orders['ROSES'].append(Order('ROSES', roses_buy, -self.position["ROSES"]))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_buy, -self.position["STRAWBERRIES"]))

        return orders
    
    def compute_etf_orders(self, order_depths: Dict[Symbol, OrderDepth], etf: OrderDepth, etf_components: Dict[Symbol, OrderDepth]) -> dict[Symbol, list[Order]]:
        synth_bid, synth_ask = self._compute_synthetic_prices(etf_components)
        self._assess_etf_arbitrage(etf, synth_bid, synth_ask)

        etf_orders = self._compute_etf_orders(order_depths)

        return etf_orders

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def vanilla_price_BS(self, S, t, K=10000, r=0., sigma=0.193329, T=250, call_put_flag='C'):
        tau = (T - t) / 365
        d1 = np.log((S * np.exp(r * tau) / K)) / (sigma * np.sqrt(tau)) + (sigma * np.sqrt(tau)) / 2
        d2 = d1 - sigma * np.sqrt(tau)
        if call_put_flag == 'C':
            vanilla_price = S * self.norm_cdf(d1) - K * np.exp(-r * tau) * self.norm_cdf(d2)
        else:
            vanilla_price = K * np.exp(-r * tau) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)

        return vanilla_price
        
    def compute_coupon_orders(self, coconuts_depth: OrderDepth, coconuts_coupon_depth: OrderDepth) -> Tuple[list[Order], list[Order]]:
        orders_coupon = []
        top_bid_coco, top_ask_coco = sorted(coconuts_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_depth.sell_orders.items())[0][0]
        top_bid_coup, top_ask_coup = sorted(coconuts_coupon_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_coupon_depth.sell_orders.items())[0][0]
        
        mid_coco = (top_bid_coco+top_ask_coco)/2
        mid_coco_coup = (top_bid_coup+top_ask_coup)/2

        imp_call_val = self.vanilla_price_BS(mid_coco, 4+(self.count/10000), K=10000, r=0, sigma=0.19332951334290502, T=250)
        
        call_spread = imp_call_val - mid_coco_coup

        logger.print(f'call_spread: {call_spread}')

        q = 30

        if call_spread < -self.coup_spread and self.position['COCONUT_COUPON'] > 0:
            orders_coupon.append(Order('COCONUT_COUPON', top_bid_coup, -self.position['COCONUT_COUPON']))

        if call_spread > self.coup_spread and self.position['COCONUT_COUPON'] < 0:
            orders_coupon.append(Order('COCONUT_COUPON', top_ask_coup, -self.position['COCONUT_COUPON']))

        if call_spread > self.coup_spread and self.position['COCONUT_COUPON'] >= 0:
            orders_coupon.append(Order('COCONUT_COUPON', top_ask_coup, q))

        if call_spread < -self.coup_spread and self.position['COCONUT_COUPON'] <= 0:
            orders_coupon.append(Order('COCONUT_COUPON', top_bid_coup, -q))

        return orders_coupon
    
    def compute_coconut_orders(self, coconuts_depth: OrderDepth, coconuts_coupon_depth: OrderDepth):
        orders_coco = []
        top_bid_coco, top_ask_coco = sorted(coconuts_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_depth.sell_orders.items())[0][0]
        top_bid_coup, top_ask_coup = sorted(coconuts_coupon_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_coupon_depth.sell_orders.items())[0][0]
        
        mid_coco = (top_bid_coco+top_ask_coco)/2
        mid_coco_coup = (top_bid_coup+top_ask_coup)/2

        option_price = self.vanilla_price_BS(mid_coco, 4+(self.count/10000), K=10000, r=0, sigma=0.19, T=250, call_put_flag='P')

        coco_spread = mid_coco_coup - option_price + 10000 - mid_coco

        q = 30

        logger.print(f'coco_spread: {coco_spread}')

        if coco_spread < -self.coco_spread and self.position['COCONUT'] > 0:
            orders_coco.append(Order('COCONUT', top_bid_coco, -self.position['COCONUT']))

        if coco_spread > self.coco_spread and self.position['COCONUT'] < 0:
            orders_coco.append(Order('COCONUT', top_ask_coco, -self.position['COCONUT']))

        if coco_spread > self.coco_spread and self.position['COCONUT'] >= 0:
            orders_coco.append(Order('COCONUT', top_ask_coco, q))

        if coco_spread < -self.coco_spread and self.position['COCONUT'] <= 0:
            orders_coco.append(Order('COCONUT', top_bid_coco, -q))

        return orders_coco

    def deserializeJson(self, json_string):
        if json_string == "":
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

        self.count += 1
        result = {}

        for key, val in state.position.items():
            self.position[key] = val

        trader_data = state.traderData
        conversion_observation = state.observations.conversionObservations
        orchid_observation = conversion_observation['ORCHIDS']
        etf_components: Dict[Symbol, OrderDepth] = {}
        etf: OrderDepth = None

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
                orchid_orders, conversions = self.compute_orchids_orders(product, order_depth, orchid_observation)
                result[product] = orchid_orders
            elif product == 'GIFT_BASKET':
                etf = order_depth
            elif product == 'CHOCOLATE' or product == 'ROSES' or product == 'STRAWBERRIES':
                etf_components[product] = order_depth
            elif product == "COCONUT":
                coconuts_depth = order_depth
            elif product == "COCONUT_COUPON":
                coconuts_coupon_depth = order_depth

        etf_orders = self.compute_etf_orders(state.order_depths, etf, etf_components)
        for prod, ords in etf_orders.items():
            if ords:
                result[prod] = ords

        result['COCONUT'] = self.compute_coconut_orders(coconuts_depth, coconuts_coupon_depth)
        result['COCONUT_COUPON'] = self.compute_coupon_orders(coconuts_depth, coconuts_coupon_depth)

        trader_data = self.serializeJson()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data