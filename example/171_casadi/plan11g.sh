cat >  /dev/shm/plan <<EOF
Ich möchte mehr über MPC lernen. Deshalb habe ich das Beispiel für invertierte Pendel gebaut. Ich verstehe nicht, welche Messdaten in das Modell einfließen.
Um eine realistischere Simulation oder ein besseres Verständnis von MPC-Controllern zu erhalten, möchte ich ein weiteres Beispiel, gen11e.lisp, erstellen.
In diesem Beispiel soll es möglich sein, die Messinstrumente auszuwählen (mittel Gui elementen), die als Eingabe für das Modell existieren. Zusätzlich soll auch eine Rauschquelle für jede dieser Eingaben mit entsprechenden GUI Widget existieren. Ist es einfach beziehungsweise möglich, diese Aenderungen am Code zu unternehmen, oder ist das sehr aufwendig?

Im existierenden MPC Modell sind auch ein paar Parameter, die ich noch nicht genau verstehe, zum Beispiel mehrere Gewichte und die Anzahl der Punkte und der Horizont. Ich verstehe nicht so richtig, wie die zwei verschiedenen Samplegrößen für die Zeit miteinander zusammenhängen. Wie muss man die Gewichte wählen, damit der Controller möglichst gut funktioniert? Genau dieselbe Frage stellt sich für den Horizont. Soll er immer so weit in die Zukunft schauen, dass er das Pendel bis zum Sollzustand bringt?
Haben die Rauschwerte auf den zu observierenden Messgrößen einen Einfluss auf diese Parameter? Bitte erkläre diese Fragen ausführlich. Ich möchte mehr verstehen, worum man sich kümmern muss, um einen MPC Controller zu bauen und was die Fallstricke dabei sind.

Als nächste Frage, möchte ich wissen, ob man auch mehrere Pendel balancieren kann und wie dies implementiert werden würde.

Eine weitere Frage bezieht sich auf mögliche Optimierungen. Kann man den Optimierer wechseln oder einen anderen angeben, um eine hoehere Performance zu erlangen? Falls das möglich ist, sollte eine entsprechende GUI Widget eingeführt werden.

Schreibe erstmal noch keinen Code, antworte erstmal auf meine Fragen.
EOF

for i in `find ../../.agents/skills/cl-py-generator/SKILL.md .agents/skills/ p11f_mpc_gui.py gen11f.lisp -type f`; do
    echo ";; start of " $i
    cat $i
done >> /dev/shm/plan
cat /dev/shm/plan | xclip
