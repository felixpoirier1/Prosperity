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
        self.roses_buy = False
        self.roses_sell = False
        self.choco_time = 0
        self.straw_time = 0

        # option
        self.coup_spread = 15
        self.coco_spread = 15
        self.count = 0
        self.day = 4
        self.last_coco_trader = ''

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
            if self.arb == 'low':
                ask_pr = mid_price-1
            elif self.arb == 'high':
                ask_pr = mid_price-1
            else:
                ask_pr = mid_price-1
        else:
            if top_ask-top_buy == 4:
                if self.arb == 'high':
                    ask_pr = top_ask-3
                elif self.arb == 'low':
                    ask_pr = top_ask-2
                else:
                    ask_pr = top_ask-3
            else:
                if second_ask-top_ask == 1:
                    ask_pr = top_ask-1
                else:
                    if self.arb == 'low':
                        ask_pr = top_ask
                    else:
                        ask_pr = top_ask-1
            
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

        if self.arb == 'high':
            if apos < 0:
                if arb_val > 0:
                    convsersions -= bpos
                    apos += convsersions
                    bpos -= convsersions
            
            if apos > -pos_lim:
                orders.append(Order(product, ask_pr, -pos_lim-apos))
                self.orch_ask = ask_pr
        
        else:
            cost_buy = ap - fee + imp
            cost_sell = bp - fee - exp

            if cost_buy < top_buy:
                quantity_to_sell = min(self.POSITION_LIMIT['ORCHIDS'] - apos, 50)
                orders.append(Order('ORCHIDS', ask_pr, -quantity_to_sell))
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
        
        synth_bid = (6*straw_buy + 4*choco_buy + rose_buy)+380
        synth_ask = (6*straw_sell + 4*choco_sell + rose_sell)+380

        return int(synth_bid), int(synth_ask)
        
    def etf_threshold(self, spread):
        if 0.5*self.spread_std <= spread < 0.75*self.spread_std:
            return 0.25
        elif 0.75*self.spread_std <= spread < 1.25*self.spread_std:
            return 0.85
        elif 1.25*self.spread_std <= spread:
            return 1
        else:
            return 0

    def _compute_etf_orders(self, order_depths: Dict[Symbol, OrderDepth], synth_bid: int, synth_ask:int, own_trades, market_trades) -> dict[Symbol, list[Order]]:
        orders: Dict[Symbol, list[Order]] = {"GIFT_BASKET": [], 'CHOCOLATE':[], 'STRAWBERRIES': [], 'ROSES':[]}

        gift_sell = sorted(order_depths["GIFT_BASKET"].sell_orders.items())[0][0]
        gift_buy = sorted(order_depths["GIFT_BASKET"].buy_orders.items(), reverse=True)[0][0]
        roses_sell = sorted(order_depths["ROSES"].sell_orders.items())[0][0]
        roses_buy = sorted(order_depths["ROSES"].buy_orders.items(), reverse=True)[0][0]
        choco_sell = sorted(order_depths["CHOCOLATE"].sell_orders.items())[0][0]
        choco_buy = sorted(order_depths["CHOCOLATE"].buy_orders.items(), reverse=True)[0][0]
        straw_sell = sorted(order_depths["STRAWBERRIES"].sell_orders.items())[0][0]
        straw_buy = sorted(order_depths["STRAWBERRIES"].buy_orders.items(), reverse=True)[0][0]

        choco_buy_qty = sorted(order_depths["CHOCOLATE"].buy_orders.items(), reverse=True)[0][1]
        choco_sell_qty = sorted(order_depths["CHOCOLATE"].sell_orders.items())[0][1]

        if synth_bid-gift_sell > 0.25*self.spread_std:
            pct_position = self.etf_threshold(synth_bid-gift_sell)

            total_qty_gift = int(pct_position*(self.POSITION_LIMIT['GIFT_BASKET']))

            if total_qty_gift > self.position['GIFT_BASKET']:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_sell, 1))

        elif gift_buy-synth_ask > 0.25*self.spread_std:
            pct_position = self.etf_threshold(gift_buy-synth_ask)

            total_qty_gift = -int(pct_position*(self.POSITION_LIMIT['GIFT_BASKET']))

            if total_qty_gift < self.position['GIFT_BASKET']:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_buy, -1))

        elif synth_bid-gift_sell < 0 and self.position['GIFT_BASKET'] > 0:
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_buy, max(-2, -self.position['GIFT_BASKET'])))

        elif gift_buy-synth_ask < 0 and self.position['GIFT_BASKET'] < 0:
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', gift_sell, min(2, -self.position['GIFT_BASKET'])))

        try:
            for i in range(len(market_trades['ROSES'])):
                if market_trades['ROSES'][i].buyer == 'Rhianna':
                    self.roses_buy = True
                    self.roses_sell = False
                elif market_trades['ROSES'][i].seller == 'Rhianna':
                    self.roses_buy = False
                    self.roses_sell = True
        except Exception as e:
            pass

        if self.roses_buy:
            orders['ROSES'].append(Order('ROSES', roses_sell, self.POSITION_LIMIT['ROSES']-self.position['ROSES']))
        elif self.roses_sell:
            orders['ROSES'].append(Order('ROSES', roses_buy, -self.POSITION_LIMIT['ROSES']-self.position['ROSES']))

        try:
            for i in range(len(market_trades['CHOCOLATE'])):
                if market_trades['CHOCOLATE'][i].buyer == 'Remy' and market_trades['CHOCOLATE'][i].seller == 'Vinnie':
                    if market_trades['CHOCOLATE'][i].timestamp != self.choco_time and market_trades['CHOCOLATE'][i].quantity == 8:
                        if -choco_sell_qty >= 50:
                            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_sell, 50))
                        else:
                            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_sell, -choco_sell_qty))
                            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_sell+1, 50+choco_sell_qty))
                        self.choco_time = market_trades['CHOCOLATE'][i].timestamp
                elif market_trades['CHOCOLATE'][i].seller == 'Remy' and market_trades['CHOCOLATE'][i].buyer == 'Vinnie':
                    if market_trades['CHOCOLATE'][i].timestamp != self.choco_time and market_trades['CHOCOLATE'][i].quantity == 8:
                        if -choco_buy_qty <= -50:
                            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_buy, -50))
                        else:
                            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_buy, -choco_buy_qty))
                            orders['CHOCOLATE'].append(Order('CHOCOLATE', choco_buy-1, -50+choco_buy_qty))
                        self.choco_time = market_trades['CHOCOLATE'][i].timestamp
            logger.print(market_trades['CHOCOLATE'])
        except Exception as e:
            pass

        try:
            for i in range(len(market_trades['STRAWBERRIES'])):
                if market_trades['STRAWBERRIES'][i].buyer == 'Remy' and market_trades['STRAWBERRIES'][i].seller == 'Vladimir':
                    if market_trades['STRAWBERRIES'][i].timestamp != self.straw_time:
                        orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_sell, 70))
                    self.straw_time = market_trades['STRAWBERRIES'][i].timestamp
                elif market_trades['STRAWBERRIES'][i].seller == 'Remy' and market_trades['STRAWBERRIES'][i].buyer == 'Vladimir':
                    if market_trades['STRAWBERRIES'][i].timestamp != self.straw_time:
                        orders['STRAWBERRIES'].append(Order('STRAWBERRIES', straw_buy, -70))
                    self.straw_time = market_trades['STRAWBERRIES'][i].timestamp
        except Exception as e:
            pass

        return orders
    
    def compute_etf_orders(self, order_depths: Dict[Symbol, OrderDepth], etf_components: Dict[Symbol, OrderDepth], own_trades, market_trades) -> dict[Symbol, list[Order]]:
        synth_bid, synth_ask = self._compute_synthetic_prices(etf_components)

        etf_orders = self._compute_etf_orders(order_depths, synth_bid, synth_ask, own_trades, market_trades)

        return etf_orders

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    def vanilla_price_BS(self, S, t, K=10000, r=0., sigma=0.1600006436832186, T=250, call_put_flag='C'):
        tau = (T - t) / 250
        d1 = np.log((S * np.exp(r * tau) / K)) / (sigma * np.sqrt(tau)) + (sigma * np.sqrt(tau)) / 2
        d2 = d1 - sigma * np.sqrt(tau)
        if call_put_flag == 'C':
            vanilla_price = S * self.norm_cdf(d1) - K * np.exp(-r * tau) * self.norm_cdf(d2)
        else:
            vanilla_price = K * np.exp(-r * tau) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)

        return vanilla_price
    
    def delta_BS(self, S, t, K=10000, r=0., sigma=0.1600006436832186, T=250):
        tau = (T - t) / 250
        d1 = np.log((S * np.exp(r * tau) / K)) / (sigma * np.sqrt(tau)) + (sigma * np.sqrt(tau)) / 2

        return self.norm_cdf(d1)
    
    def option_threshold(self, spread):
        if 5 <= spread < 10:
            return 0.2
        elif 10 <= spread < 15:
            return 0.8
        elif 15 <= spread:
            return 1
        else:
            return 0

    def compute_coupon_orders(self, coconuts_depth: OrderDepth, coconuts_coupon_depth: OrderDepth) -> Tuple[list[Order], list[Order]]:
        coup_orders = []

        top_bid_coco, top_ask_coco = sorted(coconuts_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_depth.sell_orders.items())[0][0]
        top_bid_coup, top_ask_coup = sorted(coconuts_coupon_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_coupon_depth.sell_orders.items())[0][0]

        imp_coup_bid = self.vanilla_price_BS(top_bid_coco, self.day+(self.count/10000), K=10000, r=0, sigma=0.16, T=250)
        imp_coup_ask = self.vanilla_price_BS(top_ask_coco, self.day+(self.count/10000), K=10000, r=0, sigma=0.16, T=250)

        # short VOL 
        if top_bid_coup - imp_coup_ask >= 5:
            spread = top_bid_coup - imp_coup_ask

            pct_position = self.option_threshold(spread)

            total_qty_coup = -int(pct_position*(self.POSITION_LIMIT['COCONUT_COUPON']))

            if total_qty_coup < self.position['COCONUT_COUPON']:
                coup_orders.append(Order("COCONUT_COUPON", top_bid_coup, max(-int(20*pct_position), total_qty_coup-self.position['COCONUT_COUPON'])))

        #long VOL
        elif imp_coup_bid - top_ask_coup >= 5:
            spread = imp_coup_bid - top_ask_coup

            pct_position = self.option_threshold(spread)

            total_qty_coup = int(pct_position*(self.POSITION_LIMIT['COCONUT_COUPON']))

            if total_qty_coup > self.position['COCONUT_COUPON']:
                coup_orders.append(Order("COCONUT_COUPON", top_ask_coup, min(int(20*pct_position), total_qty_coup-self.position['COCONUT_COUPON'])))

        elif top_bid_coup - imp_coup_ask < 0 and self.position["COCONUT_COUPON"] < 0:
            coup_orders.append(Order("COCONUT_COUPON", top_ask_coup, min(-self.position["COCONUT_COUPON"], 30)))

        elif imp_coup_bid - top_ask_coup < 0 and self.position["COCONUT_COUPON"] > 0:
            coup_orders.append(Order("COCONUT_COUPON", top_bid_coup, max(-self.position["COCONUT_COUPON"], -30)))

        return coup_orders
    
    def compute_coco_orders(self, coconuts_depth: OrderDepth, market_trades) -> Tuple[list[Order], list[Order]]:
        coco_orders = []

        top_bid_coco, top_ask_coco = sorted(coconuts_depth.buy_orders.items(), reverse=True)[0][0], sorted(coconuts_depth.sell_orders.items())[0][0]

        try:
            for i in range(len(market_trades['COCONUT'])):
                if market_trades['COCONUT'][i].buyer == 'Raj' and market_trades['COCONUT'][0].seller == 'Rhianna':
                    self.last_coco_trader = 'both'
                elif market_trades['COCONUT'][i].buyer == 'Raj':
                    self.last_coco_trader = 'Raj'
                if market_trades['COCONUT'][i].seller == 'Rhianna':
                    self.last_coco_trader = 'Rhianna'
        except Exception as e:
            pass    

        if self.last_coco_trader == 'both':
            if self.position['COCONUT'] > 0:
                coco_orders.append(Order('COCONUT', top_bid_coco, -self.position['COCONUT']))
            elif self.position['COCONUT'] < 0:
                coco_orders.append(Order('COCONUT', top_ask_coco, -self.position['COCONUT']))
        elif self.last_coco_trader == 'Raj':
            coco_orders.append(Order('COCONUT', top_ask_coco, min(self.POSITION_LIMIT['COCONUT']-self.position['COCONUT'], 30)))
        elif self.last_coco_trader == 'Rhianna':
            coco_orders.append(Order('COCONUT', top_bid_coco, max(-30, -self.POSITION_LIMIT['COCONUT']-self.position['COCONUT'])))

        return coco_orders

    def deserializeJson(self, json_string):
        if json_string == "":
            return
        state_dict = jsonpickle.decode(json_string)
        for key, value in state_dict.items():
            setattr(self, key, value)
    
    def serializeJson(self):
        return jsonpickle.encode(self.__dict__)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        '''
        try:
            self.deserializeJson(state.traderData)
        except:
            logger.print("Error in deserializing trader data")
        '''

        self.count += 1
        result = {}

        for key, val in state.position.items():
            self.position[key] = val

        trader_data = state.traderData
        conversions = 0
        own_trades = state.own_trades
        market_trades = state.market_trades
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

        etf_orders = self.compute_etf_orders(state.order_depths, etf_components, own_trades, market_trades)
        for prod, ords in etf_orders.items():
            if ords:
                result[prod] = ords

        result['COCONUT_COUPON'] = self.compute_coupon_orders(coconuts_depth, coconuts_coupon_depth)
        result['COCONUT'] = self.compute_coco_orders(coconuts_depth, market_trades)

        #trader_data = self.serializeJson()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data