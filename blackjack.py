from Deck import Deck
from BasicStrategy import BasicStrategy
from QLearning import QLearning

import csv

# Function to calculate the sum of card values in a hand
def hand_value(hand):
    aces = sum(1 for card in hand if card == 'A')
    value = sum(card_value(card) for card in hand)

    # Value of 'A' depending on the sum of other cards
    for _ in range(aces):
        if value + 10 <= 21:
            value += 10

    return value

# Function to value of a cards rank
def card_value(rank):
    if rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10']:
        return int(rank)
    elif rank in ['J', 'Q', 'K']:
        return 10
    elif rank == 'A':
        return 1  # Special case of 'A' handled in the hand_value function

# Function that determines the best action for a player to take based on Blackjack basic strategy
def best_action(player_hand, dealer_hand):
    strategy = BasicStrategy
    hard_totals_strategy = strategy.hard_totals_strategy
    soft_totals_strategy = strategy.soft_totals_strategy
    split_strategy = strategy.split_strategy

    # Value of the actions returned form strategies
    actions = {'H': 'HIT', 'S': 'STAND', 'D': 'DOUBLE'}
    action = 'S'    # Default action is to stand

    player_hand_value = hand_value(player_hand)
    player_card = card_value(player_hand[0])
    dealer_upcard = card_value(dealer_hand[0])

    # If dealer is showing an ace, we assume it's worth 11 for strategy
    if dealer_upcard == 1:
        dealer_upcard += 10
    
    # Firts, check if player can/should split
    if strategy.split_hand(player_hand):
        if split_strategy[player_card][dealer_upcard] == 'Y':
            return 'SPLIT'

    # Determine whether to use hard or soft strategy
    aces = sum(card == 'A' for card in player_hand)

    # Get the optimal action based on strategy
    if aces == 1 and player_hand_value < 21:
        action = soft_totals_strategy[player_hand_value][dealer_upcard]
    elif player_hand_value < 17:
        action = hard_totals_strategy[player_hand_value][dealer_upcard]

    return actions[action]

# Function that calculate the reward of a given environment
def calculate_reward(player_hand, dealer_hand):
    player_value = hand_value(player_hand)
    dealer_value = hand_value(dealer_hand)

    if player_value > 21:
        return -1       # Player busted, dealer wins
    elif dealer_value > 21:
        if player_value == 21 and len(player_hand) == 2:
            return 1    # Blackjack
        else: 
            return 1    # Dealer busted, player wins
    elif dealer_value == player_value:
        return 0    # Push
    elif player_value > dealer_value:
        if player_value == 21 and len(player_hand) == 2:
            return 1    # Blackjack
        else: 
            return 1    # Player wins
    else:
        return -1       # Dealer wins

# Function to test with just basic strategy
def blackjack_basic(num_episodes):
    wins = 0
    pushes = 0
    losses = 0
    earnings = 0

    for _ in range(num_episodes):
        
        cards = Deck
        deck = cards.deck_of_cards(1)
        player_hand = [cards.deal_card(deck), cards.deal_card(deck)]
        dealer_hand = [cards.deal_card(deck), cards.deal_card(deck)]

        wager = 100

        # List to store the players hands (incase that they split)
        all_player_hands = [player_hand]
        player_hands_after = []

        reward = 0

        while len(all_player_hands) > 0:
            
            hand = all_player_hands.pop()

            # Optimal action is to split
            if best_action(hand, dealer_hand) == 'SPLIT':

                # Split the players cards into 2 hands
                first_hand = [hand[0]]
                second_hand = [hand[1]]

                # Deal each hand a new card
                first_hand.append(cards.deal_card(deck))
                second_hand.append(cards.deal_card(deck))

                # Continue to next turn with new hands
                continue

            current_wager = wager
            # Optimal action is to hit or double down
            while best_action(hand, dealer_hand) != 'STAND' and hand_value(hand) < 21:
                action = best_action(hand, dealer_hand)

                if action == 'HIT':
                    hand.append(cards.deal_card(deck))
                elif action == 'DOUBLE':
                    current_wager = wager * 2
                    break

            player_hands_after.append(hand)

        # Optimal action is to stand and player/dealer have not busted, dealers turn
        while hand_value(dealer_hand) < 17 and hand_value(hand) <= 21:
            dealer_hand.append(cards.deal_card(deck))

        # Calculate the reward based on game outcome
        for a_hand in player_hands_after:

            if hand_value(a_hand) > 21:
                reward = -1
            elif hand_value(dealer_hand) > 21:
                reward = 1
            elif hand_value(dealer_hand) == hand_value(a_hand):
                reward = 0    # Push
            elif hand_value(a_hand) > hand_value(dealer_hand):
                reward = 1
            else:
                reward = -1   # Dealer wins
  
        # Update win, push, and loss counts
        if reward == 1:
            wins += 1
            earnings += current_wager
        elif reward == 0:
            pushes += 1
        else:
            losses += 1
            earnings -= current_wager

    print(f"Results after {num_episodes} episodes:")
    print(f"Wins: {wins} (%{wins/num_episodes * 100: .2f})")
    print(f"Pushes: {pushes} (%{pushes/num_episodes * 100: .2f})")
    print(f"Losses: {losses} (%{losses/num_episodes * 100: .2f})")
    # print(f"Earnings: ${earnings}")

# Function to test with Q-Learning from random hands
def blackjack_qlearning_1(num_episodes):
    wins = 0
    pushes = 0
    losses = 0
    qlearning = QLearning()

    for _ in range(num_episodes):

        cards = Deck
        deck = cards.deck_of_cards(1)
        agent_hand = [cards.deal_card(deck), cards.deal_card(deck)]
        dealer_hand = [cards.deal_card(deck), cards.deal_card(deck)]
        dealer_upcard = dealer_hand[0]
        dealer_upcard_value = card_value(dealer_upcard)

        # Calculate initial state and reward
        state = (hand_value(agent_hand), dealer_upcard_value)
        reward = calculate_reward(agent_hand, dealer_hand)

        next_state = state
        
        # Play the game until it reaches a terminal state
        while next_state != None:

            # Choose an action using epsilon-greedy strategy
            action = qlearning.choose_action(state)

            # Deal an additional card to the player if the action is hit
            if action == 'HIT':
                agent_hand.append(cards.deal_card(deck))
                reward = calculate_reward(agent_hand, dealer_hand)
                next_state = None if reward == -1 else (hand_value(agent_hand), card_value(dealer_hand[0]))

            else:  
                next_state = None
                
                # Deal additional cards to the dealer until their hand value is at least 17
                while hand_value(dealer_hand) < 17:
                    dealer_hand.append(cards.deal_card(deck))

                # Determine the reward based on the current state
                reward = calculate_reward(agent_hand, dealer_hand)

            # Update Q-values
            qlearning.update_q_value(state, action, reward, next_state)

            # Update state for the next iteration
            state = next_state

            # Check for terminal state (agent/player bust or blackjack)
            if hand_value(agent_hand) >= 21 or hand_value(dealer_hand) > 21:
                break

        # Update win, push, and loss counts
        if reward == 1:
            wins += 1
        elif reward == 0:
            pushes += 1
        else:
            losses += 1

    print(f"Results after {num_episodes} episodes:")
    print(f"Wins: {wins} (%{wins/num_episodes * 100: .2f})")
    print(f"Pushes: {pushes} (%{pushes/num_episodes * 100: .2f})")
    print(f"Losses: {losses} (%{losses/num_episodes * 100: .2f})")


# Fucntion to load CSV file
def load_csv(filepath):
    grid = []
    with open(filepath, 'r') as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            if(row[0] == 'PlayerNo'):
                pass
            else:
                grid.append(row)
    print('Dataset loaded!')
    return grid

# Function to train dataset
def dataset_train(dataset, ai):
    for entry in dataset:
        handsize = 1
        while(True):
            if (entry[handsize+2] != '0' and handsize+1 < 6):
                handsize+=1
                hand = []
                for i in range(handsize):
                    hand.append(entry[i+2])
                handvalue = 0
                for card in hand:
                    handvalue+=int(card)
                state = (handvalue,entry[8])

                nextState = (handvalue+int(entry[handsize+2]),int(entry[8]))

                if(entry[15] == 'Win'):
                    ai.update_q_value(state,'HIT',1,nextState)
                elif (entry[15] == 'Push'):
                    ai.update_q_value(state,'HIT',0,nextState)
                else:
                    ai.update_q_value(state,'HIT',-1,nextState)

            else:
                hand = []
                
                nextState = None
                for i in range(handsize):
                    hand.append(entry[i+2])
                
                handvalue = 0
                for card in hand:
                    handvalue+=int(card)
                state = (handvalue,int(entry[8]))
                
                if(entry[15] == 'Win'):
                    ai.update_q_value(state,'STAND',1,nextState)
                elif (entry[15] == 'Push'):
                    ai.update_q_value(state,'STAND',0,nextState)
                else:
                    ai.update_q_value(state,'STAND',-1,nextState)
                break
        

# Function to test with Q-Learning with trials from a dataset
def blackjack_qlearning_2(num_episodes):

    trials = load_csv('./blkjckhands.csv')

    blackjack = QLearning()

    dataset_train(trials,blackjack)
    print('Training complete!')

    wins = 0
    pushes = 0
    losses = 0

    for _ in range(num_episodes):

        cards = Deck
        deck = cards.deck_of_cards(1)
        player_hand = [cards.deal_card(deck), cards.deal_card(deck)]
        dealer_hand = [cards.deal_card(deck), cards.deal_card(deck)]
        player_value = hand_value(player_hand)
        dealer_upcard = dealer_hand[0]
        dealer_upcard_value = hand_value([dealer_upcard])

        # Calculate initial state and reward
        state = (hand_value(player_hand), dealer_upcard_value)
        reward = calculate_reward(player_hand, dealer_hand)

        # Play the game until it reaches a terminal state
        next_state = state
        while next_state != None:
            
            action = blackjack.choose_action(state)

            if action == 'HIT':
                player_hand.append(cards.deal_card(deck))
                player_value = hand_value(player_hand)
                if player_value > 21:
                    reward = -1
                    next_state = None
                else:
                    next_state = (player_value, hand_value([dealer_upcard]))

            elif action == 'STAND':
                next_state = None
                while hand_value(dealer_hand) < 17:
                    dealer_hand.append(cards.deal_card(deck))
                if hand_value(dealer_hand) > 21 or hand_value(player_hand) > hand_value(dealer_hand):
                    reward = 1
                elif hand_value(player_hand) == hand_value(dealer_hand):
                    reward = 0
                else:
                    reward = -1
                break

            elif action == 'SPLIT':
                # Implement split logic
                pass

            elif action == 'DOUBLE':
                # Implement double down logic
                pass
            
            blackjack.update_q_value(state, action, reward, next_state)
            state = next_state

        # Update win, push, and loss counts
        if reward == 1:
            wins += 1
        elif reward == 0:
            pushes += 1
        else:
            losses += 1

    print(f"Results after {num_episodes} episodes:")
    print(f"Wins: {wins} (%{wins/num_episodes * 100: .2f})")
    print(f"Pushes: {pushes} (%{pushes/num_episodes * 100: .2f})")
    print(f"Losses: {losses} (%{losses/num_episodes * 100: .2f})")

# Test the Blackjack reinforcement learning program

num_episodes = 10000

blackjack_basic(num_episodes)         # Test with just basic strategy
blackjack_qlearning_1(num_episodes)   # Test with Q-Learning from random hands
blackjack_qlearning_2(num_episodes)   # Test with Q-Learning from a dataset 