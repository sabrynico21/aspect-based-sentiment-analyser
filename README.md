# Aspect Based Sentiment Analysis
Il progetto svolto consiste nella creazione di un software che implementa una tecnica di analisi del testo, chiamata Aspect Based Sentiment Analysis, che consiste nell'identificare il sentimento espresso all'interno della frase rispetto ad un determinato aspetto che viene menzionato, o a cui si fa riferimento, nel testo. Per effettuare ciò si fa uso di un modello basato su BERT per identificare le porzioni di testo che fanno riferimento ad un aspetto oppure ad una opinione. Una volta ultimato il training di tale modello, viene eseguita un'analisi delle dipendenze tra le parole all'interno della frase in modo da collegare l'opinione espressa al relativo aspetto a cui si riferisce. Infine vengono valutati i risultati ottenuti sulla base di quante connessioni corrette vengono predette. 
## Classification Model
Al fine di individuare le porzioni di testo che fanno riferimento ad un aspetto oppure ad una opinione è stata implementata una classe, chiamata **ClassificationModel**, al cui interno viene utilizzato il modello BERT per ottenere, a partire dai token di una data frase, rappresentazioni che tengano conto del contesto, ovvero embeddings. Su quest'ultimi vengono applicati, separatamente, due trasformazioni lineari e successivamente la funzione softmax per ottenere rispettivamente una distribuzione di probabilità sugli aspetti ed un'altra sulle opinioni. 

Per il training del seguente modello viene utilizzato come ottimizzatore Adam e come funzione di loss la cross-entropy. A quest'ultima viene impostato il parametro 'weight' per assegnare ad ogni classe un peso proporzionale alla presenza riscontrata nel training set, in modo tale che nel calcolo della loss gli errori sulle classi meno frequenti abbiano un impatto maggiore. Sono state eseguite 50 epoche, in totale, tramite la chiamata al metodo **train_and_save**, che si occupa anche di calcolare alcune metriche, salvate in un file chiamato **classification_model_metrics**, tra cui: accuracy, precision e recall. Tali metriche vengono calcolate anche su un set di dati indipendenti dal training set per valutarne le performance.

Per ottenere le predizioni del modello è stato definito il metodo **predict** che restituisce un dizionario con due chiavi relative alle due entità da individuare: aspect e opinion. Tale metodo restituisce, a schermo, le stesse metriche calcolate durante la fase di training del modello, ed è stato utilizzato per ottenere le predizioni sul test set, le quali saranno passate alla funzione che si occupa di associare ogni opinione al rispettivo aspetto. 
### Data
È possibile scaricare il dataset utilizzato al seguente link https://www.cityscapes-dataset.com/downloads/ .
