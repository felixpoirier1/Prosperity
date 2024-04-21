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
        }

        self.long_spread = False
        self.short_spread = False
        self.exit_positions = False

    def check_position_limit(self, product, proposed_trade_size, current_position):
        limit = self.position_limits.get(product, float('inf'))
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

    def trade_starfruit2(self, state: TradingState):
        current_position = state.position.get('STARFRUIT', 0)
        orders = []

        best_bid = max(state.order_depths['STARFRUIT'].buy_orders.keys(), default=0)
        best_ask = min(state.order_depths['STARFRUIT'].sell_orders.keys(), default=0)
        spread = best_ask - best_bid

        spread_threshold = 6
        order_size_percentage_of_volume = 2

        average_bid_volume_1 = np.array([bid_volume for bid_price, bid_volume in state.order_depths['STARFRUIT'].buy_orders.items()]).mean()
        average_ask_volume_1 = np.array([ask_volume for ask_price, ask_volume in state.order_depths['STARFRUIT'].sell_orders.items()]).mean()
        order_size = min(average_bid_volume_1, average_ask_volume_1) * order_size_percentage_of_volume
        order_size = round(order_size)

        max_buy_size = self.position_limits['STARFRUIT'] - current_position
        max_sell_size = self.position_limits['STARFRUIT'] + current_position

        if spread > spread_threshold:
            direction = 'buy' if current_position <= 0 else 'sell'

            if direction == 'buy':
                if self.check_position_limit('STARFRUIT', order_size, current_position):
                    buy_order_size = min(order_size, max_buy_size)
                    if buy_order_size > 0:
                        orders.append(Order('STARFRUIT', best_bid + 1, buy_order_size))
            elif direction == 'sell':
                if self.check_position_limit('STARFRUIT', -order_size, current_position):
                    sell_order_size = min(order_size, max_sell_size)
                    if sell_order_size > 0:
                        orders.append(Order('STARFRUIT', best_ask - 1, -sell_order_size))

        return orders

    def trade_starfruit_momentum(self, state: TradingState):
        current_position = state.position.get('STARFRUIT', 0)
        orders = []

        recent_trades = state.market_trades.get('STARFRUIT', 0)
        prices = [trade.price for trade in recent_trades]

        sma_period = 10
        order_size = 3

        if len(prices) >= sma_period:
            sma = calculate_sma(prices, sma_period)
            current_price = prices[-1]

            max_buy_size = self.position_limits['STARFRUIT'] - current_position
            max_sell_size = self.position_limits['STARFRUIT'] + current_position

            if current_price > sma:
                if self.check_position_limit('STARFRUIT', order_size, current_position):
                    buy_order_size = min(order_size, max_buy_size)
                    if buy_order_size > 0:
                        orders.append(Order('STARFRUIT', current_price, buy_order_size))
            elif current_price < sma:
                if self.check_position_limit('STARFRUIT', -order_size, current_position):
                    sell_order_size = min(order_size, max_sell_size)
                    if sell_order_size > 0:
                        orders.append(Order('STARFRUIT', current_price, -sell_order_size))

        return orders

    def trade_orchids(self, state: TradingState):
        current_position = state.position.get('ORCHIDS', 0)
        orders = []
        conversions = 0
        orchid_observation = state.observations.conversionObservations['ORCHIDS']

        north_best_ask = min(state.order_depths['ORCHIDS'].sell_orders.keys(), default=float('inf'))
        north_best_bid = max(state.order_depths['ORCHIDS'].buy_orders.keys(), default=0)

        cost_to_buy_from_south = orchid_observation.askPrice + orchid_observation.transportFees + orchid_observation.importTariff
        revenue_to_sell_to_south = orchid_observation.bidPrice - orchid_observation.transportFees - orchid_observation.exportTariff

        if cost_to_buy_from_south < north_best_bid and current_position < self.position_limits['ORCHIDS']:
            quantity_to_buy = min(self.position_limits['ORCHIDS'] - current_position, 50)
            orders.append(Order('ORCHIDS', north_best_bid, -quantity_to_buy))
            conversions = quantity_to_buy

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

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        conversions = 0

        for product, order_depth in state.order_depths.items():
            orders = []

            if product == 'AMETHYSTS':
                pass
            elif product == 'STARFRUIT':
                pass
            elif product == 'ORCHIDS':
                orders, conversions = self.trade_orchids(state)

            result[product] = orders

        orders_spread = self.trade_spread(state, 80, -10)
        if orders_spread:
            result['STRAWBERRIES'] = [orders_spread['STRAWBERRIES']]
            result['CHOCOLATE'] = [orders_spread['CHOCOLATE']]
            result['ROSES'] = [orders_spread['ROSES']]
            result['GIFT_BASKET'] = [orders_spread['GIFT_BASKET']]

        traderData = "prosperity"

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
