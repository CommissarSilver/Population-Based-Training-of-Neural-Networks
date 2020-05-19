from vizdoom import *
import random
import time


# Here we create our DOOM! environment. We only use the create_environment function. The test_environment function is
# only there to make sure we've installed ViZDoom correctly and it's working.


def create_environment():
    game = DoomGame()
    game.load_config('basic.cfg')
    game.set_doom_scenario_path('basic.wad')
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()
