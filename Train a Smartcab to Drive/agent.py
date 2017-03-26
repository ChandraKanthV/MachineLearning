import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
	self.no_trials = 0
        self.waypoint=None
        self.light = None
        self.state=None
        self.testing=None


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
	self.no_trials +=1
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
	
	if testing:
		self.epsilon=0
		self.alpha=0
	else:
		#self.epsilon = 1/float(3000 - self.no_trials)

		#self.epsilon = self.epsilon - 0.05
		self.epsilon = 1/float(1+math.exp( (self.no_trials/float(1000.0)) -5.0 )) # This will give a smooth sigmoid function from 1 to 0 with about 9k trials

                #self.epsilon = math.pow(self.alpha , self.no_trials)
                #self.epsilon = 1/math.pow(self.no_trials,2)
                #self.epsilon = math.exp(-1 * self.alpha * self.no_trials/100)
                #self.epsilon = math.cos(self.alpha * self.no_trials)

		#print "\n\n\t\t\t\t\tRUNING TRIAL NUMBER:[",self.no_trials ,"] and epsilon:[", self.epsilon,"]"

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0
        
	state = None
        #state = (inputs["light"], waypoint, inputs["oncoming"], inputs["right"], inputs["left"] )
	light = inputs["light"]
        oncoming = inputs["oncoming"]
        right = inputs["right"]
        left= inputs["left"]
#	if(light == "red"):
#                if( (waypoint=="right") and (oncoming != "left") and (left != "forward")):
#                        state = (light, waypoint, waypoint)
#                else:
#                        state = (light, waypoint, None)

#        elif(light == "green"):
#                if(   (waypoint=="right") and
#                        ( (oncoming != "left" ) and (left != "forward" ) ) ):
#                                state = (light, waypoint, waypoint)

#                elif( (waypoint=="left")  and
#                        ( (oncoming == None)  and ((right == "right" or right == None ) )
#                                                and ((left == "right" or left == None ) ) ) ):
#                                state = (light, waypoint, waypoint)

#                elif( (waypoint=="forward") and
#                        ( (right == None)  and (oncoming != "left" )
#                                                and ((left == "right" or left == None ) ) ) ):
#                                state = (light, waypoint, waypoint)
#                else:
#                        state = (light, waypoint, None)

#        else:
#                state = (light, None, None)


	deadline_approaching = 100
	status = self.env.step_data
	if status.has_key('t'):
		deadline_approaching = (status["deadline"] - 1 -1) * 100.0 / (status["t"] + status["deadline"])
        state = (waypoint, inputs["light"], inputs["oncoming"], inputs["right"], inputs["left"], deadline_approaching<30 )
	self.createQ(state)
	self.waypoint = waypoint
        self.light = light
        self.state=state
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

	maxQ = None
        action_dict = self.Q[state]
        value_max = max(action_dict.values())
        items_list = action_dict.items()
        actions_list= list()
        for i,j in items_list:
                if(value_max == j):
                        actions_list.append(i)

        return random.choice(actions_list) #after few runs there will be only one element in this list

#	if not self.testing:    # during training
#		a,b,c = state
#		if c in actions_list:   #take the waypoint direction if waypoint and more actions are available
#			return c
#		elif None in actions_list:      # if waypoint is not available and None is available, take it as a 2nd priority instead of taking a risky action. 
#			return None
#		else:
#			return random.choice(actions_list) # take a random action from what ever is available
#	else:
#		return random.choice(actions_list)
#
#	return maxQ 



    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

	if ((self.learning) and ( state  not in self.Q.keys()) ):
		self.Q[state]={action:0 for action in self.valid_actions}
			#self.Q[state]={"forward":0, "right":0, "left":0, None:0} 
					#"forward":random.uniform(-0.5,0.5), 
					#"right":random.uniform(-0.05,0.05), 
					#"left":random.uniform(-0.05,0.05), 
					#None:random.uniform(-0.05,0.05)
					#}

        return
    
    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
 
	if not self.learning :
                action = random.choice(self.valid_actions)
        else:
                if self.epsilon < random.random():      # when epsilion reaching near 0
                       	action = self.get_maxQ(state)   # make learned decision
                else:
                        action = random.choice(self.valid_actions)      # make exploration

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

	if self.learning:
                self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward)
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, epsilon=1.00, alpha=0.5)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0, log_metrics=True, display=False, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=50, tolerance=0.05)


if __name__ == '__main__':
    run()
