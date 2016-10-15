import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # variables to track the reward and success of each trial
        self.trial_reward = 0
        self.reward_tracker = []
        self.trial_success = False
        self.success_tracker = []
        # initialise state
        self.state = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # store each trials reward and success
        self.reward_tracker.append(self.trial_reward)
        self.trial_reward = 0
        self.success_tracker.append(self.trial_success)
        self.trial_success = False
        # reset state
        self.state = {}

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = {"waypoint": self.next_waypoint,
                      "light": inputs["light"],
                      "oncoming": inputs["oncoming"],
                      "left": inputs["left"]}

        # TODO: Select action according to your policy
        action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        # store the reward and check for success
        self.trial_reward += reward
        if reward > 2:
            self.trial_success = True

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # remove the first item from the trackers as they are the initialised values not from a trial
    a.reward_tracker = a.reward_tracker[1:]
    a.success_tracker = a.success_tracker[1:]
    # print the reward and success trackers
    print "Reward: {}".format(a.reward_tracker)
    print "Success: {}".format(a.success_tracker)
    # calulate some statistics
    print "Number of successes: {}".format(a.success_tracker.count(True))
    print "Average reward: {}".format(np.mean(a.reward_tracker))
    # plot the rewards over time
    plt.plot(a.reward_tracker)
    plt.xlabel("Trial")
    plt.ylabel("Reward")
    plt.show()

if __name__ == '__main__':
    run()
