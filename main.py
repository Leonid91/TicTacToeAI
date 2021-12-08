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

        if len(self.availablePositions()) == 0: #S'il ne reste plus de cases à jouer
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
        result = self.winner() #On récupère le gagant
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

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

            while not self.isEnd:
                # Player 1
                positions = self.availablePositions() #On récupère les cases vides
                p1_action = self.player1.chooseAction(positions, self.board, self.playerSymbol) #On les propose au moteur et on récupère l'action
                self.updateState(p1_action) #On update le board
                board_hash = self.getHash()
                self.p1.addState(board_hash) #On enregistre le nouvel état dans l'historique des états

                win = self.winner() #On check si la partie est terminée
                if win is not None: # Si player1 win
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    #Même chose pour le joueur 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    