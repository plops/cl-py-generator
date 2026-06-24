cat <<EOF
Ich möchte mehr über MPC lernen. Deshalb habe ich das Beispiel für invertierte Pendel gebaut. Ich verstehe nicht, welche Messdaten in das Modell einfließen.
Um eine realistischere Simulation oder ein besseres Verständnis von MPC-Controllern zu erhalten, möchte ich ein weiteres Beispiel, gen11e.lisp, erstellen.
In diesem Beispiel soll es möglich sein, die Messinstrumente auszuwählen (mittel Gui elementen), die als Eingabe für das Modell existieren. Zusätzlich soll auch eine Rauschquelle für jede dieser Eingaben mit entsprechenden GUI Widget existieren. Ist es einfach beziehungsweise möglich, diese Qnderungen am Code zu unternehmen, oder ist das sehr aufwendig?
EOF > /dev/shm/plan
for i in `find ../../.agents/skills/cl-py-generator/SKILL.md .agents/skills/ p11f_mpc_gui.py gen11f.lisp -type f`; do
    echo ";; start of " $i
    cat $i
done >> /dev/shm/plan
cat /dev/shm/plan | xclip
