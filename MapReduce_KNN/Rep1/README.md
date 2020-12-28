# SpotifyAnalysis


Requirements

    mrjob 0.6.9
    numpy 1.12.1
    scipy 0.19.1

Testumgebung:

MAacOS Mojave 10.14.5 Python 3.6.4 hadoop 2.8.0

K Nächstgelegener Nachbar

KNN.py: Trainingscode für den K Nearest Neighbor-Algorithmus. Empfängt den Trainingssatz als Eingabe und gibt das Model aus

KNNPredictor.py: Code für die Vorhersage des K Nearest Neighbor-Algorithmus. Optionsmodell, das den Pfad des Mods darstellt. Empfängt den Testsatz als Eingabe und gibt die Genauigkeit aus.

Format des Datensatzes ref. /test_dataset/haberman.data.csv, jede Zeile stellt eine Probe dar, und die einzelnen Merkmale jeder Probe sind durch "," getrennt. Die letzte Spalte ist die Kategorie der Stichprobe.

python . /code/KNN.py . /test_dataset/haberman.data.csv > model.json

Ein trainiertes Modell kann erhalten werden: model.json

python . /code/KNNPredictor.py --model model.json -k 3 . /test_datensatz/haberman.test.csv

Dies kann verwendet werden, um die Leistung des Modells zu testen. Holen Sie die Ausgabe.

Accuary:73.49397590361446%

python ./MapReduce_KNN/Rep1/MRKnnSuggestion.py --model ./MapReduce_KNN/Rep1/model2.json -k 3 ./data/test_data.csv