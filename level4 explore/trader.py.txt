from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import numpy as np
import json

from math import floor, erf


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


def calculate_sma(prices, period):
    return sum(prices[-period:]) / period


def norm_cdf(x):
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


def vanilla_price_BS(S, t, K=10000, r=0., sigma=0.193329, T=250, call_put_flag='C'):
    tau = (T - t) / 365
    d1 = np.log((S * np.exp(r * tau) / K)) / (sigma * np.sqrt(tau)) + (sigma * np.sqrt(tau)) / 2
    d2 = d1 - sigma * np.sqrt(tau)
    if call_put_flag == 'C':
        vanilla_price = S * norm_cdf(d1) - K * np.exp(-r * tau) * norm_cdf(d2)
    else:
        vanilla_price = K * np.exp(-r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1)

    return vanilla_price


class Trader:

    def __init__(self):
        self.position_limits = {
            'AMETHYSTS': 20,
            'STARFRUIT': 20,
            'ORCHIDS': 100,
            'CHOCOLATE': 250,
            'STRAWBERRIES': 350,
            'ROSES': 60,
            'GIFT_BASKET': 60,
            'COCONUT': 300,
            'COCONUT_COUPON': 600,
        }

        self.long_spread = False
        self.short_spread = False
        self.exit_positions = False

    def check_position_limit(self, product, proposed_trade_size, current_position):
        limit = self.position_limits.get(product, float('inf'))  # Default to no limit if not specified
        projected_position = current_position + proposed_trade_size
        return abs(projected_position) <= limit

    def trade_amethysts(self, state: TradingState):
        current_position = state.position.get('AMETHYSTS', 0)
        orders = []

        best_bid = max(state.order_depths['AMETHYSTS'].buy_orders.keys(), default=0)
        best_ask = min(state.order_depths['AMETHYSTS'].sell_orders.keys(), default=0)

        bid_price = 9997
        ask_price = 10003

        bid_price = min(bid_price, best_ask - 1) if best_ask else bid_price
        ask_price = max(ask_price, best_bid + 1) if best_bid else ask_price

        order_size = max(abs(10000-bid_price), abs(ask_price-10000), 10)

        # Adjust order size based on position limit
        max_buy_size = self.position_limits['AMETHYSTS'] - current_position
        max_sell_size = self.position_limits['AMETHYSTS'] + current_position

        if bid_price < best_ask or not best_ask:
            if self.check_position_limit('AMETHYSTS', order_size, current_position):
                buy_order_size = min(order_size, max_buy_size)
                if buy_order_size > 0:
                    orders.append(Order('AMETHYSTS', int(bid_price), buy_order_size))

        if ask_price > best_bid or not best_bid:
            if self.check_position_limit('AMETHYSTS', -order_size, current_position):
                sell_order_size = min(order_size, max_sell_size)
                if sell_order_size > 0:
                    orders.append(Order('AMETHYSTS', int(ask_price), -sell_order_size))

        return orders

    def trade_starfruit(self, state: TradingState):
        current_position = state.position.get('STARFRUIT', 0)
        orders = []

        best_bid = max(state.order_depths['STARFRUIT'].buy_orders.keys(), default=0)
        best_ask = min(state.order_depths['STARFRUIT'].sell_orders.keys(), default=0)

        max_buy_size = self.position_limits['STARFRUIT'] - current_position
        max_sell_size = self.position_limits['STARFRUIT'] + current_position

        spread = best_ask - best_bid
        order_size = min(2, spread, max_sell_size, max_buy_size)
        short_spread = False

        if spread > 6 and not short_spread:
            if self.check_position_limit('STARFRUIT', order_size, current_position):
                orders.append(Order('STARFRUIT', best_bid+1, order_size))
                orders.append(Order('STARFRUIT', best_ask-1, -order_size))
                short_spread = True
        elif spread < 7 and short_spread:
            if self.check_position_limit('STARFRUIT', order_size, current_position):
                orders.append(Order('STARFRUIT', best_bid+1, -order_size))
                orders.append(Order('STARFRUIT', best_ask-1, order_size))
                short_spread = False

        return orders

    def trade_orchids(self, state: TradingState):
        current_position = state.position.get('ORCHIDS', 0)
        orders = []
        conversions = 0
        orchid_observation = state.observations.conversionObservations['ORCHIDS']

        # Assume North's best bid and ask prices are available in the order depth
        north_best_ask = min(state.order_depths['ORCHIDS'].sell_orders.keys(), default=float('inf'))
        north_best_bid = max(state.order_depths['ORCHIDS'].buy_orders.keys(), default=0)

        # Calculating costs and revenues for buying from and selling to the South
        cost_to_buy_from_south = orchid_observation.askPrice + orchid_observation.transportFees + orchid_observation.importTariff
        revenue_to_sell_to_south = orchid_observation.bidPrice - orchid_observation.transportFees - orchid_observation.exportTariff

        # Trading conditions
        # Buy from South if cheaper than North's best ask and sell to North
        if cost_to_buy_from_south < north_best_bid and current_position < self.position_limits['ORCHIDS']:
            quantity_to_buy = min(self.position_limits['ORCHIDS'] - current_position, 50)
            orders.append(Order('ORCHIDS', north_best_bid, -quantity_to_buy))
            conversions = quantity_to_buy

        # Sell to South if more profitable than North's best bid and there's enough stock
        if revenue_to_sell_to_south > north_best_ask:
            quantity_to_sell = min(self.position_limits['ORCHIDS'] - current_position, 50)
            orders.append(Order('ORCHIDS', north_best_ask, quantity_to_sell))
            conversions = -quantity_to_sell

        return orders, conversions

    def trade_spread(self, state: TradingState, b, c):

        STRAWBERRIES_bid, STRAWBERRIES_bid_q = list(state.order_depths['STRAWBERRIES'].buy_orders.items())[0]
        STRAWBERRIES_ask, STRAWBERRIES_ask_q = list(state.order_depths['STRAWBERRIES'].sell_orders.items())[0]

        CHOCOLATE_bid, CHOCOLATE_bid_q = list(state.order_depths['CHOCOLATE'].buy_orders.items())[0]
        CHOCOLATE_ask, CHOCOLATE_ask_q = list(state.order_depths['CHOCOLATE'].sell_orders.items())[0]

        ROSES_bid, ROSES_bid_q = list(state.order_depths['ROSES'].buy_orders.items())[0]
        ROSES_ask, ROSES_ask_q = list(state.order_depths['ROSES'].sell_orders.items())[0]

        GIFT_BASKET_bid, GIFT_BASKET_bid_q = list(state.order_depths['GIFT_BASKET'].buy_orders.items())[0]
        GIFT_BASKET_ask, GIFT_BASKET_ask_q = list(state.order_depths['GIFT_BASKET'].sell_orders.items())[0]

        prices = {
            'STRAWBERRIES': (STRAWBERRIES_bid + STRAWBERRIES_ask) / 2,
            'CHOCOLATE': (CHOCOLATE_bid + CHOCOLATE_ask) / 2,
            'ROSES': (ROSES_bid + ROSES_ask) / 2,
            'GIFT_BASKET': (GIFT_BASKET_bid + GIFT_BASKET_ask) / 2,
        }

        position_strawberries = state.position.get('STRAWBERRIES', 0)
        max_buy_strawberries = self.position_limits['STRAWBERRIES'] - position_strawberries
        max_sell_strawberries = self.position_limits['STRAWBERRIES'] + position_strawberries

        position_chocolate = state.position.get('CHOCOLATE', 0)
        max_buy_chocolate = self.position_limits['CHOCOLATE'] - position_chocolate
        max_sell_chocolate = self.position_limits['CHOCOLATE'] + position_chocolate

        position_roses = state.position.get('ROSES', 0)
        max_buy_roses = self.position_limits['ROSES'] - position_roses
        max_sell_roses = self.position_limits['ROSES'] + position_roses

        position_gift_basket = state.position.get('GIFT_BASKET', 0)
        max_buy_gift_basket = self.position_limits['GIFT_BASKET'] - position_gift_basket
        max_sell_gift_basket = self.position_limits['GIFT_BASKET'] + position_gift_basket

        spread_value = 6.1592 * prices['STRAWBERRIES'] + 3.8991 * prices['CHOCOLATE'] + 1.0368 * prices['ROSES'] - prices['GIFT_BASKET']
        orders = {}

        multiplier = 4

        STRAWBERRIES_bid, STRAWBERRIES_ask = STRAWBERRIES_bid - 1, STRAWBERRIES_ask + 1
        CHOCOLATE_bid, CHOCOLATE_ask = CHOCOLATE_bid - 1, CHOCOLATE_ask + 1
        ROSES_bid, ROSES_ask = ROSES_bid - 1, ROSES_ask + 1
        GIFT_BASKET_bid, GIFT_BASKET_ask = GIFT_BASKET_bid - 1, GIFT_BASKET_ask + 1

        can_enter = (position_strawberries == 0 and position_chocolate == 0 and position_roses == 0 and position_gift_basket == 0)

        # Exit conditions
        if spread_value < c and self.exit_positions and self.short_spread:
            # multiplier = min(floor(STRAWBERRIES_bid_q/2/6), floor(CHOCOLATE_bid_q/2/4), floor(ROSES_bid_q/2), floor(-GIFT_BASKET_ask_q/2))
            # multiplier = min(multiplier, floor(max_buy_strawberries/6), floor(max_buy_chocolate/4), max_buy_roses, max_sell_gift_basket)

            orders['STRAWBERRIES'] = Order('STRAWBERRIES', STRAWBERRIES_ask, -position_strawberries)
            orders['CHOCOLATE'] = Order('CHOCOLATE', CHOCOLATE_ask, -position_chocolate)
            orders['ROSES'] = Order('ROSES', ROSES_ask, -position_roses)
            orders['GIFT_BASKET'] = Order('GIFT_BASKET', GIFT_BASKET_bid, -position_gift_basket)

            self.exit_positions = not can_enter
            self.short_spread = not can_enter

        if spread_value > -c and self.exit_positions and self.long_spread:
            # multiplier = min(floor(-STRAWBERRIES_ask_q/2/6), floor(-CHOCOLATE_ask_q/2/4), floor(-ROSES_ask_q/2), floor(GIFT_BASKET_bid_q/2))
            # multiplier = min(multiplier, floor(max_sell_strawberries/6), floor(max_sell_chocolate/4), max_sell_roses, max_buy_gift_basket)

            orders['STRAWBERRIES'] = Order('STRAWBERRIES', STRAWBERRIES_bid, -position_strawberries)
            orders['CHOCOLATE'] = Order('CHOCOLATE', CHOCOLATE_bid, -position_chocolate)
            orders['ROSES'] = Order('ROSES', ROSES_bid, -position_roses)
            orders['GIFT_BASKET'] = Order('GIFT_BASKET', GIFT_BASKET_ask, -position_gift_basket)

            self.exit_positions = not can_enter
            self.long_spread = not can_enter

        # Entry conditions
        if spread_value > b and (not self.exit_positions or self.short_spread):
            # multiplier = min(floor(-STRAWBERRIES_ask_q/2/6), floor(-CHOCOLATE_ask_q/2/4), floor(-ROSES_ask_q/2), floor(GIFT_BASKET_bid_q/2))
            # multiplier = min(multiplier, floor(max_sell_strawberries/6), floor(max_sell_chocolate/4), max_sell_roses, max_buy_gift_basket)

            orders['STRAWBERRIES'] = Order('STRAWBERRIES', STRAWBERRIES_bid, -6 * multiplier)
            orders['CHOCOLATE'] = Order('CHOCOLATE', CHOCOLATE_bid, -4 * multiplier)
            orders['ROSES'] = Order('ROSES', ROSES_bid, -1 * multiplier)
            orders['GIFT_BASKET'] = Order('GIFT_BASKET', GIFT_BASKET_ask, 1 * multiplier)

            self.exit_positions = True
            self.short_spread = True

        if spread_value < -b and (not self.exit_positions or self.long_spread):
            # multiplier = min(floor(STRAWBERRIES_bid_q/2/6), floor(CHOCOLATE_bid_q/2/4), floor(ROSES_bid_q/2), floor(-GIFT_BASKET_ask_q/2))
            # multiplier = min(multiplier, floor(max_buy_strawberries/6), floor(max_buy_chocolate/4), max_buy_roses, max_sell_gift_basket)

            orders['STRAWBERRIES'] = Order('STRAWBERRIES', STRAWBERRIES_ask, 6 * multiplier)
            orders['CHOCOLATE'] = Order('CHOCOLATE', CHOCOLATE_ask, 4 * multiplier)
            orders['ROSES'] = Order('ROSES', ROSES_ask, 1 * multiplier)
            orders['GIFT_BASKET'] = Order('GIFT_BASKET', GIFT_BASKET_bid, -1 * multiplier)

            self.exit_positions = True
            self.long_spread = True

        return orders

    def trade_option(self, state: TradingState, b, c):

        COCONUT_bid = max(state.order_depths['COCONUT'].buy_orders.keys(), default=0)
        COCONUT_ask = min(state.order_depths['COCONUT'].sell_orders.keys(), default=0)

        COCONUT_COUPON_bid = max(state.order_depths['COCONUT_COUPON'].buy_orders.keys(), default=0)
        COCONUT_COUPON_ask = min(state.order_depths['COCONUT_COUPON'].sell_orders.keys(), default=0)

        mid_price_COCONUT = (COCONUT_bid + COCONUT_ask) / 2
        mid_price_COCONUT_COUPON = (COCONUT_COUPON_bid + COCONUT_COUPON_ask) / 2

        # COCONUT_COUPON_bid, COCONUT_COUPON_ask = COCONUT_COUPON_bid - 1, COCONUT_COUPON_ask + 1

        position_coconut = state.position.get('COCONUT', 0)

        position_coconut_coupon = state.position.get('COCONUT_COUPON', 0)

        # option_price = vanilla_price_BS(mid_price_COCONUT, state.timestamp/1000000 + 3, K=10000, r=0., sigma=0.19, T=250, call_put_flag='C')
        # option_price = vanilla_price_BS(mid_price_COCONUT, state.timestamp / 1000000 + 4, K=10000, r=0., sigma=0.19, T=250, call_put_flag='C')
        option_price = vanilla_price_BS(mid_price_COCONUT, 4, K=10000, r=0., sigma=0.19, T=250, call_put_flag='C')

        spread_value = option_price - mid_price_COCONUT_COUPON
        orders = []

        q = 30

        # Exit conditions
        if spread_value < c and position_coconut_coupon > 0:
            orders.append(Order('COCONUT_COUPON', COCONUT_COUPON_bid, -position_coconut_coupon))

        if spread_value > -c and position_coconut_coupon < 0:
            orders.append(Order('COCONUT_COUPON', COCONUT_COUPON_ask, -position_coconut_coupon))

        # Entry conditions
        if spread_value > b and position_coconut_coupon >= 0:
            # positive: I think the option is underpriced, so I want to buy it
            orders.append(Order('COCONUT_COUPON', COCONUT_COUPON_ask, q))

        if spread_value < -b and position_coconut_coupon <= 0:
            # negative: I think the option is overpriced, so I want to sell it
            orders.append(Order('COCONUT_COUPON', COCONUT_COUPON_bid, -q))

        return orders

    def trade_coconuts(self, state: TradingState, b, c):

        COCONUT_bid = max(state.order_depths['COCONUT'].buy_orders.keys(), default=0)
        COCONUT_ask = min(state.order_depths['COCONUT'].sell_orders.keys(), default=0)

        COCONUT_COUPON_bid = max(state.order_depths['COCONUT_COUPON'].buy_orders.keys(), default=0)
        COCONUT_COUPON_ask = min(state.order_depths['COCONUT_COUPON'].sell_orders.keys(), default=0)

        mid_price_COCONUT = (COCONUT_bid + COCONUT_ask) / 2
        mid_price_COCONUT_COUPON = (COCONUT_COUPON_bid + COCONUT_COUPON_ask) / 2

        # COCONUT_COUPON_bid, COCONUT_COUPON_ask = COCONUT_COUPON_bid - 1, COCONUT_COUPON_ask + 1

        position_coconut = state.position.get('COCONUT', 0)

        position_coconut_coupon = state.position.get('COCONUT_COUPON', 0)

        # option_price = vanilla_price_BS(mid_price_COCONUT, state.timestamp/1000000 + 3, K=10000, r=0., sigma=0.19, T=250, call_put_flag='P')
        # option_price = vanilla_price_BS(mid_price_COCONUT, state.timestamp / 1000000 + 4, K=10000, r=0., sigma=0.19, T=250, call_put_flag='P')
        option_price = vanilla_price_BS(mid_price_COCONUT, 4, K=10000, r=0., sigma=0.19, T=250, call_put_flag='P')

        spread_value = mid_price_COCONUT_COUPON - option_price + 10000 - mid_price_COCONUT
        orders = []

        q = 30

        # Exit conditions
        if spread_value < c and position_coconut > 0:
            orders.append(Order('COCONUT', COCONUT_bid, -position_coconut))

        if spread_value > -c and position_coconut < 0:
            orders.append(Order('COCONUT', COCONUT_ask, -position_coconut))

        # Entry conditions
        if spread_value > b and position_coconut >= 0:
            # positive: I think the option is underpriced, so I want to buy it
            orders.append(Order('COCONUT', COCONUT_ask, q))

        if spread_value < -b and position_coconut <= 0:
            # negative: I think the option is overpriced, so I want to sell it
            orders.append(Order('COCONUT', COCONUT_bid, -q))

        return orders

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        conversions = 0

        for product, order_depth in state.order_depths.items():
            orders = []

            if product == 'AMETHYSTS':
                orders = self.trade_amethysts(state)
            elif product == 'STARFRUIT':
                orders = self.trade_starfruit(state)
            elif product == 'ORCHIDS':
                orders, conversions = self.trade_orchids(state)
            elif product == 'COCONUT':
                orders = self.trade_coconuts(state, 10, -10)
            elif product == 'COCONUT_COUPON':  # STRIKE PRICE: 10000 - EXPIRY: 250
                orders = self.trade_option(state, 2, -2)

            result[product] = orders

        orders_spread = self.trade_spread(state, 80, -10)
        if orders_spread:
            result['STRAWBERRIES'] = [orders_spread['STRAWBERRIES']]
            result['CHOCOLATE'] = [orders_spread['CHOCOLATE']]
            result['ROSES'] = [orders_spread['ROSES']]
            result['GIFT_BASKET'] = [orders_spread['GIFT_BASKET']]

        traderData = "(F)RED_ISLAND"

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
