import numpy as np
import pickle
import keras
from keras.models import Sequential, save_model, load_model

BOARD_ROWS = 3
BOARD_COLS = 3

#Un state est un état unique qui est défini par la position des symboles des deux joueurs sur le board
class State:

    #Constructeur de State
    def __init__(self, p1, p2): 
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS)) #On initialise le board avec des 0 de partout
        self.player1 = p1
        self.player2 = p2
        self.isFinished = False #Sert à savoir si la partie est terminée
        self.boardHash = None #Sert à identifier l'état de la partie

        #Le joueur 1 place des 1; le joueur 2 place des -1 dans les matrices
        #Le joueur 1 commence
        self.playerSymbol = 1 

    #Permet d'obtenir un identifiant unique de l'état en fonction des cases qui ont été jouées par les deux joueurs
    def hashBoard(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
    
    #Permet de savoir si la partie est terminée, et qui est le gagnant ou s'il y a une égalité
    def isGameFinished(self):
        #Rappel :
        # 0  si la case est vide
        # 1  si p1 a joué la case
        # -1 si p2 a joué la case
        for i in range(BOARD_ROWS): #Pour chaque ligne
            if sum(self.board[i, :]) == 3: #On teste si la somme de la ligne égale 3
                self.isFinished = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isFinished = True
                return -1

        for i in range(BOARD_COLS): #Pour chaque colonne
            if sum(self.board[:, i]) == 3:
                self.isFinished = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isFinished = True
                return -1

        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)]) #Somme de {(0,0); (1,1); (2,2)}
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)]) #Somme de {(0,2); (1,2); (2,1)}
        diag_sum = max(abs(diag_sum1), abs(diag_sum2)) #On prend le max de la valeur absolue deux sommes et on regarde si c'est égal à 3 pour savoir si quelqu'un a gagné

        if diag_sum == 3: #Si quelqu'un a gagné
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        if len(self.getAvailablePositions()) == 0: #S'il ne reste plus de cases à jouer
            self.isFinished #La partie est terminée
            return 0 #Egalité
        
        #Sinon la partie n'est pas terminée
        self.isEnd = False
        return None

    #Permet d'obtenir les cases qui sont encore vides dans le board
    def getAvailablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    #Met à jour le board et change de joueur
    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    #Assigne des "rewards" aux 2 joueurs
    def giveReward(self):
        result = self.isGameFinished() #On récupère le gagant
        if result == 1:
            self.player1.feedReward(1)
            self.player2.feedReward(0)
        elif result == -1:
            self.player1.feedReward(0)
            self.player2.feedReward(1)
        else:
            self.player1.feedReward(0.1)
            self.player2.feedReward(0.5)

    #Réinitialise le board aux valeurs par défaut
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
    
    #Logique d'entraintement en Reinforcement Learning
    def trainRL(self, rounds=100):
        #rounds = nombre d'entrainements (100 par défaut)
        for i in range(rounds): 
            if i % 1000 == 0: #Tous les multiples de 1000
                print("Rounds {}".format(i)) #On affiche la progression

            while not self.isFinished:
                # Player 1
                positions = self.getAvailablePositions() #On récupère les cases vides
                p1_action = self.player1.chooseAction(positions, self.board, self.playerSymbol) #On les propose au moteur et on récupère l'action
                self.updateState(p1_action) #On update le board
                board_hash = self.hashBoard()
                self.player1.addState(board_hash) #On enregistre le nouvel état dans l'historique des états

                win = self.isGameFinished() #On check si la partie est terminée
                if win is not None: # Si player1 win
                    self.giveReward()
                    self.player1.reset()
                    self.player2.reset()
                    self.reset()
                    break

                else:
                    #Même chose pour le joueur 2
                    positions = self.getAvailablePositions()
                    p2_action = self.player2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.hashBoard()
                    self.player2.addState(board_hash)

                    win = self.isGameFinished()
                    if win is not None:
                        self.giveReward()
                        self.player1.reset()
                        self.player2.reset()
                        self.reset()
                        break

    #Jeu contre un adversaire (joueur ou autre IA)
    def play(self):
        # On fait pareil mais sans de logique d'entrainement
        while not self.isFinished:
            # Player 1 (IA)
            positions = self.getAvailablePositions()
            p1_action = self.player1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(p1_action)
            self.showBoard() #Affichage du board
            win = self.isGameFinished()
            if win is not None:
                if win == 1:
                    print(self.player1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.getAvailablePositions()
                p2_action = self.player2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.isGameFinished()
                if win is not None:
                    if win == -1:
                        print(self.player2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

class Player:
    def __init__(self, name, exp_rate=0.3): 
        self.name = name
        self.states = []  #Historique de tous les états
        self.lr = 0.2 #Learning Rate
        self.exp_rate = exp_rate #Exploration rate (pour explorer d'autres choix)
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    # Logique de choix d'action pour un état donné
    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate: # On fait un rand
            idx = np.random.choice(len(positions)) # On choisit aléatoirement une case
            action = positions[idx]
        else:
            value_max = -999
            for p in positions: #On parcourt toutes les positions possibles
                next_board = current_board.copy() # La prochain état correspond à l'état de ce board
                next_board[p] = symbol # avec le symbole à la nouvelle position
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash) # On compare les valeurs de chaque board
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self, state):
        self.states.append(state)

    # Attribution des rewards par backpropagation dans tous les états joués
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    # Permet de sauvegarder le modèle (tous les scores pour chaque état)
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    # Chargement d'un modèle
    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass

class ANNPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, current_board, symbol):
        while True:
            reshapedBoard = current_board.reshape(BOARD_COLS * BOARD_ROWS) # On met le board au bon format pour l'input de l'ANN
            convertedBoard = []
            toPredict = []
            for val in reshapedBoard: # On transforme les valeurs pour correspondre à celles connues par l'ANN
                if val == 1:
                    convertedBoard.append(2)
                elif val == -1 :
                    convertedBoard.append(1)
                else :
                    convertedBoard.append(0)

            for pos in positions:
                index1D = (3*pos[0]) + pos[1] # On calcule l'index en 1 dim
                #boardToPredict = np.empty_like(convertedBoard)
                #boardToPredict = convertedBoard
                copy = np.copy(convertedBoard)
                boardToPredict = copy.tolist()
                if symbol == 1:
                    boardToPredict[index1D] = 2 #On remplace le symbole de la position dispo par le symbole de l'IA
                elif symbol == -1:
                    boardToPredict[index1D] = 1 #On remplace le symbole de la position dispo par le symbole de l'IA
                
                toPredict.append(boardToPredict)

            predictions = (model.predict(toPredict) > 0.5).astype(int)
            i = 0
            action = (-1,-1)
            for pred in predictions: # On parcourt toutes les prédictions
                for val in pred:
                    if val == 1:
                        action = positions[i] # On choisit la première prédiction = 1
                if action == (-1,-1):
                    i = i+1
                else :
                    break

            if action in positions:
                return action
            else:
                return positions[0] # Si il n'y a aucune prédiction à 1, on joue la première position

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    print(" (1) -- Train the AI with reinforcement")
    print(" (2) -- Play against the reinforced learning trained AI")
    print(" (3) -- Play against the ANN trained AI")
    print(" (4) -- Watch a game between both AI")
    print(" (0) -- Leave")
    decision  = int(input("Input your action : "))

    # Reinforcement learning training
    if decision == 1:
        p1 = Player("p1")
        p2 = Player("p2")

        st = State(p1, p2)
        print("training...")
        st.trainRL(50000)
        p1.savePolicy()

    elif decision == 2:
        p1 = Player("RL", exp_rate=0)
        p1.loadPolicy("policy_p1")

        p2 = HumanPlayer("human")

        st = State(p1, p2)
        st.play()

    elif decision == 3:
        p1 = ANNPlayer("ANN")
        model = load_model('./saved_ann_model', compile = True)

        p2 = HumanPlayer("human")

        st = State(p1, p2)
        st.play()           

        
