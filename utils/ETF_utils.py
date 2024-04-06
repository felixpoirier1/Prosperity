"""
These are helper functions I wrote for the RIT competition to help with trading ETFs. This is not directly applicable to Prosperity.

If we get a round like last year:
    ETF := PICNIC_BASKET
    ETF \approx 2*BAGUETTE + 4*DIP + UKULELE + 375

Then we can use the utilities to create a "Synthetic ETF" from the underlyings and assess different metrics such as liquidity, optimal price, etc.
"""

import pandas as pd
import timeit
import numpy as np

def enoughLiquidty(book: dict[pd.DataFrame, pd.DataFrame], slack: float, quantity: int, price: float, direction: str) -> bool:
    """Determines if there is enough liquidity in the order book to fill a given order

    Parameters
    ----------
    book : pd.DataFrame
        dict of pd.DataFrames for bids and asks
    slack : float
        the slack to add to the quantity to be filled 
    quantity : int
        the quantity to be filled
    price : float
        the price at which the order is to be filled at minimum
    direction : str
        the direction of the order, either "BUY" or "SELL"

    Returns
    -------
    bool
        True if there is enough liquidity, False otherwise

    Raises
    ------
    ValueError
        if direction is not "BUY" or "SELL"
    """
    if direction == "SELL":
        book = book["asks"]
        book = book.loc[book["price"] >= price]
        quantity_book = (book["quantity"] - book["quantity_filled"])*(1+slack)


        if quantity_book.sum() >= quantity:
            return True
        else:
            return False

    elif direction == "BUY":
        book = book["bids"]
        book = book.loc[book["price"] <= price]
        quantity_book = (book["quantity"] - book["quantity_filled"])*(1+slack)

        if quantity_book.sum() >= quantity:
            return True
        else:
            return False
        
    else:
        raise ValueError("Direction must be either BUY or SELL")
    
def optimalPrice(book: dict[pd.DataFrame, pd.DataFrame], quantity: int, direction: str) -> float:
    """Determines the optimal price to fill a given quantity

    Parameters
    ----------
    book : dict[pd.DataFrame, pd.DataFrame]
        dict of pd.DataFrames for bids and asks
    quantity : int
        the quantity to be filled
    direction : str
        the direction of the order, either "BUY" or "SELL"

    Returns
    -------
    float
        the optimal price to fill the given quantity

    Raises
    ------
    ValueError
        if direction is not "BUY" or "SELL"
    """
    if direction == "SELL":
        book = book["asks"]
        quantity_book = book["quantity"] - book["quantity_filled"]
        quantity_book = quantity_book.cumsum()
        price = book.loc[quantity_book >= quantity, "price"].iloc[0]
        return price

    elif direction == "BUY":
        book = book["bids"]
        quantity_book = book["quantity"] - book["quantity_filled"]
        quantity_book = quantity_book.cumsum()
        price = book.loc[quantity_book >= quantity, "price"].iloc[0]
        return price
        
    else:
        raise ValueError("Direction must be either BUY or SELL")
    
import pandas as pd

def createSyntheticETF(books: dict[str, dict[np.ndarray, np.ndarray]]):
    """
    This function takes books for two securities and creates a synthetic book for an ETF consisting of the two securities

    Parameters
    ----------
    books : dict[str, dict[np.ndarray, np.ndarray]]
        dict of books for two securities

    Returns
    -------
    np.ndarray
        The book with the proper formatting for comparison to the ETF
    """

    # initialize the synthetic book
    synth_book = []

    # create a list of the two book names
    books_name = list(books.keys())

    # keep iterating until there are no more orders in either book
    while books[books_name[0]].shape[0] > 0 and books[books_name[1]].shape[0] > 0:
        # find the lowest volume for the first orders on the books
        min_quantity = min(books[books_name[0]][0][1], books[books_name[1]][0][1])

        # calculate the price of the synthetic order
        price = books[books_name[0]][0][0] + books[books_name[1]][0][0]

        # create the new order
        synth_book.append([min_quantity, price])

        # subtract the quantity from both books
        books[books_name[0]][0][1] -= min_quantity
        books[books_name[1]][0][1] -= min_quantity

        # remove the order from the book if it's completely filled
        if books[books_name[0]][0][1] == 0:
            books[books_name[0]] = np.delete(books[books_name[0]], 0, axis=0)

        if books[books_name[1]][0][1] == 0:
            books[books_name[1]] = np.delete(books[books_name[1]], 0, axis=0)

    return np.array(synth_book)



def findOptimalArbitrageQty(bidbook : np.ndarray, askbook : np.ndarray, slack: float = 0.02) -> int:
    """_summary_

    Parameters
    ----------
    bidbook : np.ndarray
        The bid side of the book for the security that is overvalued
    askbook : np.ndarray
        The ask side of the book for the security that is undervalued
    slack : float, optional
        Quantity to add or remove from the price to account for commission and slippage, by default 0.02

    Returns
    -------
    int
        the optimal quantity to buy/sell in the ETF/ synthetic security
    """
    bidbook[:, 1] = bidbook[:, 1] - slack
    askbook[:, 1] = askbook[:, 1] + slack

    bidbook_ = bidbook[bidbook[:, 1] > askbook[0, 1]]
    askbook_ = askbook[askbook[:, 1] < bidbook[0, 1]]

    qty = min(bidbook_[:, 0].sum(), askbook_[:, 0].sum())

    return round(qty / 10) * 10

def findVwap(book : np.ndarray, direction : str, qty : int, slack : float = 0.05) -> float:
    """This function takes a book as a parameter and returns the VWAP for the book given a certain quantity

    Parameters
    ----------
    book : np.ndarray
        The book to calculate the VWAP for
    direction : str
        "bid" or "ask"
    slack : float, optional
        price to subtract from vwap to account for trading fees including slippage, by default 0.05

    Returns
    -------
    _type_
        _description_
    """
    #only keep orders on the book where the cumulative quantity is less than the quantity we want to trade and include outstanding volume from last order
    if direction == "bid":
        book = book[book[:, 0].cumsum() <= qty]
 
    elif direction == "ask":
        book = book[book[:, 0].cumsum() <= qty]
    
    #calculate the VWAP
    if direction == "bid":
        vwap = (book[:, 0] * (book[:, 1] - slack)).sum() / book[:, 0].sum()
    elif direction == "ask":
        vwap = (book[:, 0] * (book[:, 1] + slack)).sum() / book[:, 0].sum()
    
    return vwap
    
def isProfitable(price : int, book : np.ndarray, direction : str, qty : int, slack : float = 0.05) -> bool:
    """This function takes a book and a price and determines if it is profitable to trade at that price

    Parameters
    ----------
    price : int
        The price to check
    book : np.ndarray
        The book to check
    direction : str
        The direction of the order, either "bid" or "ask"
    slack : float, optional
        price to subtract from vwap to account for trading fees including slippage, by default 0.05

    Returns
    -------
    bool
        True if profitable, False if not
    """
    vwap = findVwap(book, direction, qty, slack)
    if direction == "bid":
        return vwap > price
    elif direction == "ask":
        return vwap < price


if __name__ == "__main__":
    #create BULL book
    BULL = {
        "bids": np.array(pd.DataFrame({
            "price": [101, 100, 99, 97, 96, 95, 94, 93, 92, 91],
            "quantity": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "quantity_filled": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })), 
        "asks": np.array(pd.DataFrame({
            "price": [99, 101, 103, 104, 105, 106, 107, 108, 109, 110],
            "quantity": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "quantity_filled": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }))

    }   

    #create BEAR book
    BEAR = {
        "bids": np.array(pd.DataFrame({
            "price": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            "quantity": [130, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        })),     
        "asks": np.array(pd.DataFrame({
            "price": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "quantity": [120, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        }))  
    }
    print(BEAR["asks"])

    #print(createSyntheticETF({"security1": BULL, "security2": BEAR}, "asks"))
        
    def TESTcreateSyntheticETF():
        print(createSyntheticETF({"security1": BULL, "security2": BEAR}))
        findOptimalArbitrageQty(BULL["bids"], BULL["asks"], 5)
        return None
    
    #print(timeit.timeit(TESTcreateSyntheticETF, number=1000))

    def TESTisProfitable():
        isProfitable(BULL["bids"], "bids", 5)
        return None
    
    print(isProfitable(99, BEAR["bids"], "bid", 500))
