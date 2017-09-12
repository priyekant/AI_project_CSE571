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
import random, util, sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        currentFood = currentGameState.getFood();
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        """
        here we are going to update the score value on the basis of the distance
        from the ghost, capsules and food
        """
        score = 0.0
        """check distance from old capsule positions to know if the capsule has been eaten or not"""
        for capsuleposition in currentGameState.getCapsules():
            capsuledistance = manhattanDistance(capsuleposition, newPos)
            if capsuledistance == 0:
                """add 10 points if the new position is equal to capsule position in current state"""
                score += 10.0
            else:
                """points are added to the total score, closer the capsule more the points encouraging pacman to take that path"""
                score += 10.0/capsuledistance

        """check distance from newghoststates as ghosts are also moving"""
        for ghost in newGhostStates:
            ghostdistance = manhattanDistance(ghost.getPosition(), newPos)
            if ghostdistance <= 1:
                if(ghost.scaredTimer > 0):
                    """200 points are awarded if pacman eats ghost"""
                    score += 200.0
                else:
                    """500 points are lost if ghost eats pacman"""
                    score -= 500.0
            else:
                if(ghost.scaredTimer > 0):
                    score += 100.0/ghostdistance
        numFood = currentGameState.getNumFood()
        """check distance from old capsule positions to know if the capsule has been eaten or not"""
        for x in range(currentFood.width):
          for y in range(currentFood.height):
            if(currentFood[x][y]):
              fooddistance=manhattanDistance(newPos,(x,y))
              if(fooddistance==0):
                """10 points are added to the score if pacman eats a food after takin this action"""
                score+=10.0
                """if in current state only one food was left and after taking this action pacman ate that food then add 500 points"""
                if numFood == 1:
                    score += 500.0
              else:
                """points are added to the total score, closer the food more the points encouraging pacman to take that path"""
                score+=1.0/fooddistance
        return score

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
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        scores = [self.minvalue(self.depth,gameState.generateSuccessor(0,action), 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def maxvalue(self,depth,gameState):
        """we check for terminal test condition, either game is over or we have
        traversed the whole depth"""
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        scores = [self.minvalue(depth,gameState.generateSuccessor(0,action), 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return bestScore,legalMoves[chosenIndex]

    def minvalue(self,depth, gameState, agentnumber):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentnumber)
        """the last agent will again call for maxvalue instead of minvalue as it is now turn of the pacman who is a maxagent"""
        if(agentnumber < gameState.getNumAgents() - 1):
            scores = [self.minvalue(depth,gameState.generateSuccessor(agentnumber,action), agentnumber+1) for action in legalMoves]
        else:
            scores = [self.maxvalue(depth-1, gameState.generateSuccessor(agentnumber,action)) for action in legalMoves]
        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score,move = self.maxalphabetavalue(self.depth,gameState,-sys.maxint,sys.maxint)

    def maxalphabetavalue(self,depth,gameState,alphavalue,betavalue):
        """we check for terminal test condition, either game is over or we have
        traversed the whole depth"""
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(0)

        score = -sys.maxint
        index = 0
        for action in legalMoves:
            score = max(score,self.minalphabetavalue(depth,gameState.generateSuccessor(0,action), 1,alphavalue,betavalue))
            if score >= betavalue:
                return score,legalMoves[index]
            alphavalue = max(alphavalue,score)
            index += 1

        return score,legalMoves[index-1]

    def minalphabetavalue(self,depth,gameState,agentnumber,alphavalue,betavalue):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentnumber)
        score = sys.maxint
        """the last agent will again call for maxvalue instead of minvalue as it is now turn of the pacman who is a maxagent"""

        for action in legalMoves:
            if(agentnumber < gameState.getNumAgents() - 1):
                score = min(score,self.minalphabetavalue(depth,gameState.generateSuccessor(agentnumber,action), agentnumber+1,alphavalue,betavalue))
            else:
                score = min(score,self.maxalphabetavalue(depth-1,gameState.generateSuccessor(agentnumber,action),alphavalue,betavalue))
            if score <= alphavalue:
                return score
            betavalue = min(betavalue,score)

        return score


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
