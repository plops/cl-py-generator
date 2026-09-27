for i in plan/20260924_01_cluster_search/walkthrough_de.md untersuchung_technisch_de.md loader.py sweep_umap.py build_cluster_data.py validate.py cluster_search.py best_params_compact.json plot_final.py
do
    echo "// start of "$i
    cat $i
done


echo "mache einen review dieses python projekts. koennte man die auswertung/ analyse robuster machen, wenn man design of experiments anwendet? wie wuerde so etwas aussehen?"

#echo "als ein deliverable moechte ich mithilfe der untersuchung ein rust programm bauen, dass neue embeddings in der bestehenden datenbank einordnet. das soll dazu dienen um bei einem neu ermittelten summary das zugehoerige cluster zu finden (falls es eins gibt) und die nahegelegensten punkte im latenten embedding raum aufzulisten. dieses rust programm soll mit moeglichst wenig abhaengigkeiten und z.b. ohne gpu auskommen. das umap mapping ist recht aufwaendig zu bestimmen (ich wuerde es vielleicht alle 2 monate mit dem python framework regenerieren lassen. wobei auch da die bestimmung der cluster titel sehr aufwaendig ist (denn ein LLM muss die titel von beispielen des clusters in hinblick auf nachbar cluster erzeugen). es waere gut dafuer eine art online update mechanismus einzusetzen, wo nur neue samples ins llm geschickt werden, falls sich cluster wesentlich aendern (neue hinzukommen)."
