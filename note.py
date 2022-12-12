from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array

def game_list(_):
    games_list = pyspiel.registered_games()
    print("Registered games:")
    for game in games_list:
        print(" ", game.short_name)
    print()

def count_repetitive_plays_in_history(history):
    hashmap = {}
    for i in history.split(" "):
        prec_value = 0 if hashmap.get(i) == None else hashmap.get(i)
        hashmap[i] = prec_value + 1
    print(hashmap.values())

def example_api_of_games(_):
    print(pyspiel.registered_names())
    game = pyspiel.load_game("tic_tac_toe")
    # Print name of the game: tic_tac_toe()
    print(game)
    # Print max and min utility possible
    print(game.max_utility(), game.min_utility())
    # Print max number of actions
    print(game.num_distinct_actions())

    # Create a state of the game
    state = game.new_initial_state()
    # Print it to see visually
    print(state)
    # Who's the player playing at this state
    print(state.current_player())
    # Is the state a terminal one ? One winner
    print(state.is_terminal())
    # Accumulated reward to all players currently
    print(state.returns())
    # All possible actions: Between 0 and num_distinct_actions()-1
    print(state.legal_actions())
    # apply the action on the state
    state.apply_action(1)

    print("Other Game")
    # Can parametrize the game:
    game = pyspiel.load_game("breakthrough(rows=6,columns=6)")
    # Create a state of the game
    state = game.new_initial_state()
    print(state)
    for action in state.legal_actions():
        print("{} {}".format(action, state.action_to_string(action)))
    state.apply_action(112)
    print(state)

    print("Other games")
    # Create the game matching pennies, automatically generated and same api
    game = pyspiel.create_matrix_game([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    state = game.new_initial_state()
    print(state)
    # This will print -2 because it is in simultaneous play
    state.current_player()
    # action of player 0
    state.legal_actions(0)
    # Apply TWO actions bc simultaneous: After that, it's terminal and returns are [1, -1]
    state.apply_actions([0, 0])

    print("Other example")
    # Dynamics API
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_matrix = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
    # Define what the population playes generally with proba
    x = np.array([0.2, 0.2, 0.6])
    # The gradient showed in video
    print(dyn(x))
    # Define a step functions
    alpha = 0.01
    # Manually run
    x += alpha * dyn(x)
    # Multiple steps
    for i in range(10):
        x += alpha * dyn(x)
    print(x)
    return "example finished"

def gameTesting():
    game = pyspiel.load_game("yorktown")
    state = game.new_initial_state()
    print("Initial state:\n{}".format(state))
    for i in range(10):
        print(state.legal_actions())
        print(state.legal_actions()[0], state.action_to_string(state.current_player(), state.legal_actions()[0]))
        state.apply_action(state.legal_actions()[0])
    print(state)
    print(state.information_state_string())
    state.apply_action(state.legal_actions()[0])
    print(state.information_state_string())
    print(state.history())
    # Seems unusable
    # print(state.information_state_tensor())
    # Not implemented
    # print(state.observation_string())
    # print(state.observation_tensor())

def evaluatebot():
    # solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="yorktown"))
    # result = pyspiel.evaluate_bots(game.new_initial_state(), bots, seed=0)
    return 