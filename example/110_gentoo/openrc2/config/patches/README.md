# Quell-Patches für Sanitizer-Treffer

Regel: Schlägt ein Sanitizer im Boot-Pfad an, wird der **gemeldete Bug in der
Quelle gefixt**, nicht der Sanitizer abgestellt. Patches liegen hier als
`config/patches/<kategorie>/<paket>/NNN-<name>.patch` und werden vom Dockerfile
nach `/etc/portage/patches/` kopiert — Portage wendet sie automatisch an.

Jeder Patch braucht im `walkthrough.md` einen Eintrag: ASan-Report (gekürzt),
Ursache, Fix-Idee, Boot-Beleg nach Rebuild. `nosanitizer`-Ausnahmen nur mit
Begründung und nur als letzte Option.
