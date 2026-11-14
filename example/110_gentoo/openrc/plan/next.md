i really need vfat support (to develop pi pico 2)

Es gibt keinen 100% standardmäßigen "Drop-in"-Ersatz für /bin/bash oder /usr/bin/sh direkt von Werk aus, der vollständig binär- und skriptkompatibel ist und die System-Shells blind ersetzen kann. Das liegt daran, dass das POSIX-Verhalten und jahrzehntelange Bash-Spezifika (Bashismen) extrem komplex nachzubauen sind.
Es gibt jedoch zwei sehr vielversprechende Open-Source-Projekte, die genau dieses Ziel verfolgen, sowie mächtige Alternativen, um Bash-Skripte durch Rust zu ersetzen:

1. Echte Versuche eines Drop-in-Ersatzes (Shell-Kompatibilität)

• Brush: Dies ist aktuell der ambitionierteste Versuch einer Bash-kompatiblen Shell in Rust. Er lädt deine bestehende .bashrc, unterstützt Aliase, Funktionen und Bash-Vervollständigungen. Er kann als interaktive Shell genutzt oder als Rust-Crate eingebettet werden. [1] (https://brush.sh/)
• Rush: Eine zu ~96% POSIX-sh-kompatible Shell, die in Rust geschrieben ist. Sie bietet einen Skript-Modus zur Ausführung von .sh-Dateien sowie Pipes, Weiterleitungen und Standard-Builtins. [1] (https://crates.io/crates/rush-sh)
Hinweis: Es wird dringend davon abgeraten, /bin/sh oder /bin/bash auf Systemebene (z.B. für Systemd-Boot-Skripte) hart durch diese zu ersetzen, da subtile Inkompatibilitäten das System unbrauchbar machen können.

2. Rust als direkter Ersatz für Shell-Skripte (Scripting-Ansatz)

Wenn du Bash-Skripte loswerden und stattdessen die Performance und Sicherheit von Rust nutzen willst, gibt es hervorragende Tools, die sich wie Shell-Skripte anfühlen:
• rust-script: Ermöglicht es dir, einzelne .rs-Dateien direkt auszuführen, ohne manuell cargo new oder cargo build auszuführen. Du kannst sogar einen Shebang am Anfang deiner Rust-Datei nutzen: [1] (https://www.kurtlawrence.info/blog/gufdjkjkq7wphfhkvrumcvmqdr4r69)
rust
#!/usr/bin/env rust-script
fn main() {
    println!("Hallo aus Rust, das sich wie ein Bash-Skript verhält!");
}
Use code with caution.
• Nushell (nu): Keine direkte Bash-Kopie, sondern eine moderne, in Rust geschriebene Shell. Sie arbeitet mit strukturierten Daten (Tabellen/Objekten) anstelle von purem Text-Piping und ist die populärste "Rust-Shell" für den Alltag. [1] (https://www.reddit.com/r/rust/comments/11athq7/oxidise_your_life/)

3. Moderne Rust-Alternativen für Core-Utilities

Falls du nicht die Shell selbst, sondern die typischen GNU-Werkzeuge (ls, cat, grep) durch moderne, schnelle Rust-Varianten ersetzen möchtest, gibt es etablierte Drop-in-Alternativen, die du einfach per Alias in deiner Bash hinterlegen kannst: [1] (https://www.youtube.com/watch?v=rWMQ-g2QDsI)
• grep → ripgrep (rg) (extrem schnell)
• cat → bat (mit Syntax-Highlighting)
• ls → eza (moderner Nachfolger von exa)
• find → fd
• cd → zoxide (lernt deine Pfade) [1] (https://dev.to/dev_tips/15-rust-cli-tools-that-will-make-you-abandon-bash-scripts-forever-4mgi), [2] (https://www.youtube.com/watch?v=rWMQ-g2QDsI)
Möchtest du ein bestehendes Bash-Skript nach Rust portieren und suchst nach passenden Crates für Befehlsausführungen, oder suchst du primär eine neue interaktive Shell für dein Terminal?
