import random

class Deck:
    # Function to initialze a 52 card deck
    def deck_of_cards(number_of_decks):
        deck = (4 * ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']) * number_of_decks
        random.shuffle(deck)

        return deck

    # Function to deal a card from a given deck
    def deal_card(deck):
        if not deck:
            raise ValueError("The deck is empty")
        card = deck.pop()

        return card