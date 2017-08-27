# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    """
    "*** YOUR CODE HERE ***"

    actionsList = []
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    visited.add(problem.getStartState())
    level = 1
    successors = problem.getSuccessors(problem.getStartState())
    myStack = util.Stack()
    for succ in successors:
        succList = list(succ)
        succList.append(level)
        succ = tuple(succList)
        myStack.push(succ)
    reachedLeaf = True
    while myStack.isEmpty() == False:
        nextNode = myStack.pop()
        if reachedLeaf == True:
            while len(actionsList) >= nextNode[3]:
                actionsList.pop()
        actionsList.append(nextNode[1])
        if problem.isGoalState(nextNode[0]):
            print "Goal state found"
            return actionsList
        visited.add(nextNode[0])
        successors = problem.getSuccessors(nextNode[0])
        for succ in successors:
            if succ not in myStack.list and succ[0] not in visited:
                succList = list(succ)
                succList.append(nextNode[3]+1)
                succ = tuple(succList)
                myStack.push(succ)
                readchedLeaf = False


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actionsList = []
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    frontier = list()
    visited.add(problem.getStartState())
    successors = problem.getSuccessors(problem.getStartState())
    myQueue = util.Queue()
    pathQueue = util.Queue()
    for succ in successors:
        if succ not in frontier and succ[0] not in visited:
            pathList = list()
            pathList.insert(len(pathList),succ[1])
            pathQueue.push(pathList)
            myQueue.push(succ)
            frontier.insert(len(frontier),succ[0])

    while myQueue.isEmpty() == False:
        nextNode = myQueue.pop()
        nextPath = pathQueue.pop()
        visited.add(nextNode[0])
        if problem.isGoalState(nextNode[0]):
            pathList = list(nextPath)
            return pathList
        successors = problem.getSuccessors(nextNode[0])
        for succ in successors:
            if succ[0] not in frontier and succ[0] not in visited:
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue.push(pathList)
                myQueue.push(succ)
                frontier.insert(len(frontier),succ[0])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    actionsList = []
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    frontier = list()
    visited.add(problem.getStartState())
    successors = problem.getSuccessors(problem.getStartState())
    myQueue = util.PriorityQueue()
    pathQueue = util.PriorityQueue()
    for succ in successors:
        if succ not in frontier and succ[0] not in visited:
            pathList = list()
            pathList.insert(len(pathList),succ[1])
            pathQueue.push(pathList,succ[2])
            myQueue.push(succ,succ[2])
            frontier.insert(len(frontier),succ[0])

    while myQueue.isEmpty() == False:
        nextNode = myQueue.pop()
        nextPath = pathQueue.pop()
        visited.add(nextNode[0])
        if problem.isGoalState(nextNode[0]):
            pathList = list(nextPath)
            return pathList
        successors = problem.getSuccessors(nextNode[0])
        for succ in successors:
            if succ[0] not in frontier and succ[0] not in visited:
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue.push(pathList,succ[2])
                myQueue.push(succ,succ[2])
                frontier.insert(len(frontier),succ[0])

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
