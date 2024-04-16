import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict, Tuple
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
        self.POSITION_LIMIT = {'STARFRUIT':20, 'AMETHYSTS':20, 'ORCHIDS':100, 'CHOCOLATE': 250, 'ROSES': 60, "GIFT_BASKET": 60 , 'STRAWBERRY': 350}
        self.sf_params = [0.08442609, 0.18264657, 0.7329293]

        # orchids
        self.orch_ask = 0
        self.orch_bid = 0
        self.arb = None
        self.etf_norm_const = 1000 #the lower the value, the more liquidity taking is prioritized over market making

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
        bpos = self.position[product]
        convsersions = 0

        ap = observation.askPrice
        imp = observation.importTariff
        fee = observation.transportFees

        top_ask, top_buy = sorted(order_depth.sell_orders.items())[0][0], sorted(order_depth.buy_orders.items())[-1][0]
        second_ask = sorted(order_depth.sell_orders.items())[0][0]
        ask_pr = self.get_orchid_prices(-imp-fee, top_ask, top_buy, second_ask)

        ask_arb_val = max(ask_pr, self.orch_ask)

        arb_val = ask_arb_val-ap-imp-fee

        if apos < 0:
            if arb_val > 0:
                convsersions -= bpos
                apos += convsersions
                bpos -= convsersions
            else:
                if self.arb == 'low':
                    orders.append(Order(product, max(self.orch_ask-1, int((top_ask+top_buy)/2))+1, -bpos))
                    bpos -= bpos

        if apos > -pos_lim:
            orders.append(Order(product, ask_pr, -pos_lim-apos))
            self.orch_ask = ask_pr

        return orders, convsersions

    @staticmethod
    def _compute_synthetic_etf(etf_components: Dict[Symbol, OrderDepth]) -> OrderDepth:
        """This computes the synthetic ETF from the ETF components (which allows us to express these products in terms of the ETF)

        Args:
            order_depth (Dict[Symbol, OrderDepth]): The order depth of the ETF components

        Returns:
            OrderDepth: A dictionary
        """
        synth_book = {}
        # seperate order depths for each component and side
        strawberries_buy, strawberries_sell = list(etf_components["STRAWBERRIES"].buy_orders.items()), list(etf_components["STRAWBERRIES"].sell_orders.items())
        chocolate_buy, chocolate_sell = list(etf_components["CHOCOLATE"].buy_orders.items()), list(etf_components["CHOCOLATE"].sell_orders.items())
        roses_buy, roses_sell = list(etf_components["ROSES"].buy_orders.items()), list(etf_components["ROSES"].sell_orders.items())
        synth_buy: OrderDepth = OrderDepth()

        #compute bid side of synth book
        while (strawberries_buy and chocolate_buy and roses_buy):
            # first level is always equivalent to product with lowest quantity
            min_qty = min(strawberries_buy[0][1], chocolate_buy[0][1], roses_buy[0][1])
            price = strawberries_buy[0][0] + 4*chocolate_buy[0][0] + 6*roses_buy[0][0] + 400
            if strawberries_buy[0][1] == min_qty:
                strawberries_buy.pop(0)
            if chocolate_buy[0][1] == min_qty:
                chocolate_buy.pop(0)
            if roses_buy[0][1] == min_qty:
                roses_buy.pop(0)
            synth_buy.buy_orders[price] = min_qty

        #compute ask side of synth book
        while (strawberries_sell and chocolate_sell and roses_sell):
            min_qty = min(strawberries_sell[0][1], chocolate_sell[0][1], roses_sell[0][1])
            price = strawberries_sell[0][0] + 4*chocolate_sell[0][0] + 6*roses_sell[0][0] + 400
            if strawberries_sell[0][1] == min_qty:
                strawberries_sell.pop(0)
            if chocolate_sell[0][1] == min_qty:
                chocolate_sell.pop(0)
            if roses_sell[0][1] == min_qty:
                roses_sell.pop(0)
            synth_buy.sell_orders[price] = min_qty

        return synth_book
    
    def _func_mm_lt_tradeoff(self, spread: float, net_position: int) -> float:
        """
        :math MM(pos, spread) = 
        \begin{cases}
            \exp(-\frac{pos^2 + spread^2}{b}) & \text{if } spread < 0 \\
            1                                 & \text{otherwise}
        \end{cases}
        """
        return np.exp(-((net_position**2 + spread**2)/self.etf_norm_const))

    def _func_qty_to_trade(self, spread: float) -> float:
        """
        1 = \exp(c * cutoff_spread) -1 => c = \frac{ln(2)}{cutoff_spread}

        this comes from,
        f(x) = \exp(cutoff_spread*x) - 1 if x < cutoff_spread else 1

        the reason for this is that we are exponentially increasing the qty to trade as the spread increases, 
        feel free to play with cutoff_spread to see how it affects the trading strategy
        """
        return 1 if spread > 80 else np.exp(spread*0.008664) - 1
        
        
    def _assess_etf_arbitrage(self, etf: OrderDepth, synth_etf: OrderDepth) -> Tuple[int, int]:
        """this computes the optimal qty to buy and sell for the ETF both for market making and liquidity taking
        """
        side = None
        # when the ETF is trading at a discount
        if etf.sell_orders[0][0] < synth_etf.buy_orders[0][0]:
            side = "undervalued"
            # 1. Assess share of MM vs LT
            if self.etf_pos <= 0:
                mm_pct = 1
            else:
                mm_pct = self._func_mm_lt_tradeoff(synth_etf.buy_orders[0][0] - etf.sell_orders[0][0], self.etf_pos)
            
            # 2. Compute the optimal qty to buy & sell (should we slowly build up the position or go all in?)
            # this could possibly be improved
            qty_to_buy = min(
                #maybe we should be fine with diverging from the spread a bit
                min(
                    abs(self.position["STRAWBERRY"]) - self.POSITION_LIMIT["STRAWBERRY"], 
                    abs(self.position["CHOCOLATE"]) - self.POSITION_LIMIT["CHOCOLATE"], 
                    abs(self.position["ROSES"]) - self.POSITION_LIMIT["ROSES"], 
                    abs(self.POSITION_LIMIT["GIFT_BASKET"]) - self.position["GIFT_BASKET"]
                ),
                abs(self._func_qty_to_trade(synth_etf.buy_orders[0][0] - etf.sell_orders[0][0])), 
                min(etf.sell_orders[0][1], synth_etf.buy_orders[0][1])
                )
            qty_to_mm = int(mm_pct * qty_to_buy)
            qty_to_lt = qty_to_buy - qty_to_mm

        # when the ETF is trading at a premium
        elif etf.sell_orders[0][0] > synth_etf.buy_orders[0][0]:
            side = "overvalued"
            # 1. Assess share of MM vs LT
            if self.etf_pos >= 0:
                mm_pct = 1
            else:
                mm_pct = self._func_mm_lt_tradeoff(etf.sell_orders[0][0] - synth_etf.buy_orders[0][0], self.etf_pos)
            
            # 2. Compute the optimal qty to buy & sell (should we slowly build up the position or go all in?)
            qty_to_sell = min(
                #maybe we should be fine with diverging from the spread a bit
                min(
                    ((abs(self.position["STRAWBERRY"]) - self.POSITION_LIMIT["STRAWBERRY"])//6, 
                    ((abs(self.position["CHOCOLATE"]) - self.POSITION_LIMIT["CHOCOLATE"]))//4, 
                    abs(self.position["ROSES"]) - self.POSITION_LIMIT["ROSES"], 
                    abs(self.POSITION_LIMIT["GIFT_BASKET"]) - self.position["GIFT_BASKET"]
                ),
                abs(self._func_qty_to_trade(etf.sell_orders[0][0] - synth_etf.buy_orders[0][0])), 
                min(etf.sell_orders[0][1], synth_etf.buy_orders[0][1])
                )
            )
            qty_to_mm = int(mm_pct * qty_to_sell)
            qty_to_lt = qty_to_sell - qty_to_mm
        else:
            return 0, 0
        
        return qty_to_mm, qty_to_lt if side == "undervalued" else -qty_to_mm, -qty_to_lt
        
    def _compute_etf_mm_orders(self, etf: OrderDepth, synth_etf: OrderDepth, qty_to_mm: int) -> dict[Symbol, list[Order]]:
        """this computes and returns the orders for market making
        """
        orders = {}

        # we want to buy
        orders["GIFT_BASKET"] = Order("GIFT_BASKET", synth_etf.buy_orders[0][0], qty_to_mm)
        orders["STAWBERRY"] = Order("STRAWBERRY", synth_etf.sell_orders[0][0], -6*qty_to_mm)
        orders["CHOCOLATE"] = Order("CHOCOLATE", synth_etf.sell_orders[0][0], -4*qty_to_mm)
        orders["ROSES"] = Order("ROSES", synth_etf.sell_orders[0][0], -qty_to_mm)
        return orders
        

    def _compute_etf_lt_orders(self, etf: OrderDepth, synth_etf: OrderDepth, qty_to_lt: int) -> dict[Symbol, list[Order]]:
        """this computes and returns the orders for liquidity taking
        """
        orders = []

        # we want to buy
        orders["GIFT_BASKET"] = Order("GIFT_BASKET", etf.sell_orders[0][0], qty_to_lt)
        orders["STRAWBERRY"] = Order("STRAWBERRY", synth_etf.sell_orders[0][0], -6*qty_to_lt)
        orders["CHOCOLATE"] = Order("CHOCOLATE", synth_etf.sell_orders[0][0], -4*qty_to_lt)
        orders["ROSES"] = Order("ROSES", synth_etf.sell_orders[0][0], -qty_to_lt)
        return orders
    
    def compute_etf_orders(self, etf: OrderDepth, etf_components: Dict[Symbol, OrderDepth]) -> dict[Symbol, list[Order]]:
        """ This method computes the optimal order for the ETF and it's underlying products
        :math GIFT_BASKET = 1*ROSES + 4*CHOCOLATE + 6*STRAWBERRY + 400

        Args:
            etf (OrderDepth): OrderDepth of the ETF
            etf_components (Dict[Symbol, OrderDepth]): OrderDepth of the ETF components

        Returns:
            dict[Symbol, list[Order]]: A dictionary containing the orders for the ETF components in this order (ROSES, CHOCOLATE, STRAWBERRY, GIFT_BASKET)
        """
        synth_book = self._compute_synthetic_etf(etf_components)
        qty_to_mm, qty_to_lt = self._assess_etf_arbitrage(etf, synth_book)
        if qty_to_mm == 0 and qty_to_lt == 0:
            return {} 
        # compute the orders for both market making and liquidity taking
        mm_orders = self._compute_etf_mm_orders(etf, synth_book, qty_to_mm)
        lt_orders = self._compute_etf_lt_orders(etf, synth_book, qty_to_lt)

        all_orders = {}
        # merge the orders
        for asset in ["ROSES", "CHOCOLATE", "STRAWBERRY", "GIFT_BASKET"]:
            all_orders[asset] = []
            if asset in lt_orders:
                all_orders[asset].extend(mm_orders[asset])
            if asset in mm_orders:
                mm_orders[asset].extend(lt_orders[asset])

        return mm_orders





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
            elif product == 'CHOCOLATE' or product == 'ROSES' or product == 'STRAWBERRY':
                etf_components[product] = order_depth

        self.compute_etf_orders(etf_components)

        trader_data = self.serializeJson()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data