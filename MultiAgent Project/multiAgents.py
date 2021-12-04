# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        food_dist = math.inf
        newGhostPositions = successorGameState.getGhostPositions()
        
        for ghost in newGhostPositions:
            ghost_dist = manhattanDistance(newPos, ghost)
            if (ghost_dist < 2):
                return -1
        
        for food in newFoodList:
            actual_dist = manhattanDistance(newPos, food)
            food_dist = min(food_dist,actual_dist)
        
        # Evaluation function
        food_dist_reverse = 1.0/food_dist
        func_val = successorGameState.getScore() + food_dist_reverse
        return func_val

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #eval function value at root node - pacman
        retur_val = self.maxim(gameState, 0, 0)
        return retur_val[0]     #returns action

    def maxim(self, gameState, depth, agent_index):
        agent_num = gameState.getNumAgents()
        compare_val = ("", -math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num   #for pacman, index will be always Zero 
            minimax_val = self.processminimax(successor_state, i, agent_ind)
            
            #checks for the maximum value
            if minimax_val > compare_val[1]:
                maxim_val = (act, minimax_val)
                compare_val = maxim_val
            else:
                maxim_val = compare_val
                
        return maxim_val

    def minim(self, gameState, depth, agent_index):
        
        agent_num = gameState.getNumAgents()
        compare_val = ("", math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num       #for pacman, index will be always Zero 
            minimax_val = self.processminimax(successor_state, i, agent_ind)
            
            #checks for the minimum value
            if minimax_val < compare_val[1]:
                minim_val = (act, minimax_val)
                compare_val = minim_val
            else:
                minim_val = compare_val
        
        return minim_val
    
    def processminimax(self, gameState, depth, agent_index):
        agent_num = gameState.getNumAgents()
        #depth == 2 indicates two times of all the agents
        if gameState.isLose() or gameState.isWin() or depth == (agent_num * self.depth):
            return self.evaluationFunction(gameState)
        
        if agent_index == 0:
            #pacman eval function values
            maxim = self.maxim(gameState, depth, agent_index)
            return maxim[1]
        else:
            #ghost eval function values
            minim = self.minim(gameState, depth, agent_index)
            return minim[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #eval function value at root node - pacman
        retur_val = self.maxim(gameState, 0, 0, -math.inf, math.inf)
        return retur_val[0]     #returns action

    def maxim(self, gameState, depth, agent_index, a, b):
        agent_num = gameState.getNumAgents()
        compare_val = ("", -math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num   #for pacman, index will be always Zero 
            processalphabeta_val = self.processalphabeta(successor_state, i, agent_ind, a, b)
            
            #checks for the maximum value
            if processalphabeta_val > compare_val[1]:
                maxim_val = (act, processalphabeta_val)
                compare_val = maxim_val
            else:
                maxim_val = compare_val
                
            if maxim_val[1] > b: 
                return maxim_val
            else: 
                a = max(a, maxim_val[1])
                
        return maxim_val

    def minim(self, gameState, depth, agent_index, a, b):
        
        agent_num = gameState.getNumAgents()
        compare_val = ("", math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num   #for pacman, index will be always Zero 
            processalphabeta_val = self.processalphabeta(successor_state, i, agent_ind, a, b)
            
            #checks for the minimum value
            if processalphabeta_val < compare_val[1]:
                minim_val = (act, processalphabeta_val)
                compare_val = minim_val
            else:
                minim_val = compare_val
                
            if minim_val[1] < a: 
                return minim_val
            else: 
                b = min(b, minim_val[1])
        
        return minim_val
    
    def processalphabeta(self, gameState, depth, agent_index, a, b):
        agent_num = gameState.getNumAgents()
        #depth == 2 indicates two times of all the agents
        if gameState.isLose() or gameState.isWin() or depth == (agent_num * self.depth):
            return self.evaluationFunction(gameState)
        
        if agent_index == 0:
            #pacman eval function values
            maxim = self.maxim(gameState, depth, agent_index, a, b)
            return maxim[1]
        else:
            #ghost eval function values
            minim = self.minim(gameState, depth, agent_index, a, b)
            return minim[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
    
        agent_num = gameState.getNumAgents()
        total_depth = agent_num * self.depth
        retur_val = self.maxim(gameState, total_depth, 0, "changenodeaction")
        return retur_val[0]     #returns action

    def maxim(self, gameState, depth, agent_index, act):
        agent_num = gameState.getNumAgents()
        compare_val = ("", -math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for acts in legal_actions:
            action_successor = None
            agent_ind = (agent_index + 1) % agent_num   #for pacman, index will be always Zero
            i = depth - 1
            successor_state = gameState.generateSuccessor(agent_index, acts)
            if depth != agent_num * self.depth:
                action_successor = act
            else:
                action_successor = acts
            expectimax_val = self.processexpectimax(successor_state, i, agent_ind, action_successor)
            
            #checks for the maximum value
            if expectimax_val > compare_val[1]:
                maxim_val = (acts, expectimax_val)
                compare_val = maxim_val
            else:
                maxim_val = compare_val
                
        return maxim_val

    def expect(self, gameState, depth, agent_index, act):
        
        agent_num = gameState.getNumAgents()
        legal_actions = gameState.getLegalActions(agent_index)
        propability_val = 1.0/len(legal_actions)
        expect_val = 0
        for acts in legal_actions:
            i = depth - 1
            successor_state = gameState.generateSuccessor(agent_index, acts)
            agent_ind = (agent_index + 1) % agent_num      #for pacman, index will be always Zero 
            expectimax_val = self.processexpectimax(successor_state, i, agent_ind, act)
            
            expect_val += expectimax_val * propability_val

        return (act,expect_val)
    
    def processexpectimax(self, gameState, depth, agent_index, act):
        #depth == 2 indicates two times of all the agents
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
        
        if agent_index == 0:
            #pacman eval function values
            maxim = self.maxim(gameState, depth, agent_index, act)
            return maxim[1]
        else:
            #ghost eval function values
            expect = self.expect(gameState, depth, agent_index, act)
            return expect[1]

