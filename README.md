# Aspect Based Sentiment Analysis
Il progetto svolto consiste nella creazione di un software che implementa una tecnica di analisi del testo, chiamata Aspect Based Sentiment Analysis, che consiste nell'identificare il sentimento espresso all'interno della frase rispetto ad un determinato aspetto che viene menzionato, o a cui si fa riferimento, nel testo. Per effettuare ciò si fa uso di un modello basato su BERT per identificare le porzioni di testo che fanno riferimento ad un aspetto oppure ad una opinione. Una volta ultimato il training di tale modello, viene eseguita un'analisi delle dipendenze tra le parole all'interno della frase in modo da collegare l'opinione espressa al relativo aspetto a cui si riferisce. Infine vengono valutati i risultati ottenuti sulla base di quante connessioni corrette vengono predette. 
## Classification Model
Al fine di individuare le porzioni di testo che fanno riferimento ad un aspetto oppure ad una opinione è stata implementata una classe, chiamata **ClassificationModel**, al cui interno viene utilizzato il modello BERT per ottenere, a partire dai token di una data frase, rappresentazioni che tengano conto del contesto, ovvero embeddings. Su quest'ultimi vengono applicati, separatamente, due trasformazioni lineari e successivamente la funzione softmax per ottenere rispettivamente una distribuzione di probabilità sugli aspetti ed un'altra sulle opinioni. 

Per il training del seguente modello viene utilizzato come ottimizzatore Adam e come funzione di loss la cross-entropy. A quest'ultima viene impostato il parametro 'weight' per assegnare ad ogni classe un peso proporzionale alla presenza riscontrata nel training set, in modo tale che nel calcolo della loss gli errori sulle classi meno frequenti abbiano un impatto maggiore. Sono state eseguite 50 epoche, in totale, tramite la chiamata al metodo **train_and_save**, che si occupa anche di calcolare alcune metriche, salvate in un file chiamato **classification_model_metrics**, tra cui: accuracy, precision e recall. Tali metriche vengono calcolate anche su un set di dati indipendenti dal training set per valutarne le performance.

Per ottenere le predizioni del modello è stato definito il metodo **predict** che restituisce un dizionario con due chiavi relative alle due entità da individuare: aspect e opinion. Tale metodo restituisce, a schermo, le stesse metriche calcolate durante la fase di training del modello, ed è stato utilizzato per ottenere le predizioni sul test set, le quali saranno passate alla funzione che si occupa di associare ogni opinione al rispettivo aspetto. 
## Analisi delle dipendenze
Una volta ottenuto un array che identifica la presenza di una certa entità nel testo, sia per gli aspetti che per le opinioni, si fa uso della funzione **get_predicted_labels** per ottenere per ciascuna frase un array di oggetti **Label**; classe a cui vengono impostate le seguenti proprietà: 'aspect_text', 'opinion_text', 'polarity', 'category_aspect' e 'category_opinion'. 

Tale funzione utilizza la libreria SpaCy per ottenere l'albero che presenta le dipendenze tra le parole. Fatto ciò, per ogni opinione riscontrata in una determinata frase si va ad identificare in che posizione si trova in modo tale da accedere al corrispettivo token restituito effettuando il parsing della frase e poter accedere alle rispettive informazioni. Viene utilizzato l'attributo 'head' per spostarsi sulla parola padre fino a quando non si riscontra un match con un aspetto, oppure fin che il valore di 'head' coincide col token corrente. In quest'ultimo caso verranno analizzati i figli della 'head', usando l'attributo 'children'. Se dopo tali analisi non si è trovato l'aspetto corrispondente, verrà assegnato ad esso il valore "NULL" e la rispettiva categoria col valore "RESTAURANT". Durante tale processo, si tiene conto degli aspetti che sono stati già assegnati.

Se vi sono aspetti che non sono stati accoppiati con l'analisi precedentemente effettuata, si andrà ad iterare su di essi per verificare che siano effettivamente presenti nel testo, e in tal caso verrà aggiunta una **Label** con il testo e la categoria relativa all'aspetto, con l'attributo che contiene il testo dell'opinione col valore "NULL", mentre la sua categoia col valore "GENERAL".

Infine, se non si è trovata alcuna opinione o aspetto nel testo, verra aggiuntà un'unica **Label** che presenta il valore "NULL" per entrambe le entità e come categoria verrà assegnata, per l'aspetto "RESTAURANT", mentre per l'opinione "MISCELLANEOUS".
## Metriche adottate
Per valutare le predizioni finali, viene definita una funzione, chiamata **get_performances**, che per ogni **Label** di una determinata frase va a individuare tra le predizioni del modello quale si la Label il cui testo, relativo sia all'aspetto che all'opinione, abbia un Intersection Over Union (IoU) maggiore. Una volta ottenuto il match, per ogni coppia vengono calcolate metriche specifiche per ogni attributo in **Label**; in particolare, per 'aspect_text' e 'opinion_text' vengono calcolati i rispettivi IoU score, mentre per 'polarity', 'category_aspect' e 'category_opinion' viene calcolata l'accuracy. Viene fatta una media di queste metriche per le coppie di una frase e successivamente una media di tale media per tutte le frasi. 

Inoltre, viene calcolata la recall, andando a verificare, data una frase, quante corrispondenze con le label originali si hanno rispetto alla cardinalità di quest'ultime, e la precision, dividendo il precedente numeratore per la cardinalità delle label predette per una frase. Come prima si fa una media di tali metriche calcolare su ogni esempio.
## Utilizzo del software
L'esecuzione del software può avvenire in due modalità la cui scelta avviene inserendo da riga di comando l'argomento --mode seguito da 'train' oppure 'test'. La prima modalità andrà ad effettuare semplicemente il training del modello **ClassificationModel**, ovvero la chiamata al metodo **train_and_save**. Mentre, la modalità di test prende in input un ulteriore argomento, ovvero --model, a cui bisogna passare il path del modello già addestrato in modo tale da poter ottenere le predizioni ed andare a restituire le metriche desiderate.
