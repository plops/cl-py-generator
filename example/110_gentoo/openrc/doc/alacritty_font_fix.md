# Alacritty Font Fix Walkthrough (E14)

This walkthrough records how the bad-looking Alacritty font on the E14
(`kiel@e14`, Alacritty 0.16.1, X11 via `startx` + dwm) was diagnosed and fixed
on 2026-11-19.

## Symptom

Text in Alacritty looked wrong: uneven glyph widths, poor spacing, generally
"not a terminal font".

## Diagnosis

### 1. Check which font Alacritty asks for

```bash
cat ~/.config/alacritty/alacritty.toml
```

The config requested `Unifont` for normal, bold and italic:

```toml
[font.normal]
family = "Unifont"
style = "Regular"
```

### 2. Check what fontconfig actually resolves

```bash
fc-match Unifont
fc-match "Unifont:bold"
```

Result:

```text
LiberationSans-Regular.ttf: "Liberation Sans" "Regular"
LiberationSans-Bold.ttf: "Liberation Sans" "Bold"
```

Unifont was **not installed**. Fontconfig silently fell back to
**Liberation Sans**, a proportional font. That is the cause of the ugly
rendering.

### 3. Check which fonts exist at all

```bash
fc-list : family | sort -u
```

Only Liberation (Mono/Sans/Serif), Noto Emoji and WenQuanYi Zen Hei were
installed. The image's `config/var/lib/portage/world` contains no dedicated
monospace font for terminals.

### Why not just install Unifont?

- `media-fonts/unifont` without USE=`ttf` installs only PCF bitmap fonts.
- The enabled fontconfig rule `70-no-bitmaps-except-emoji.conf` rejects
  bitmap fonts, so Alacritty would still not find it.
- USE=`ttf` pulls in `media-gfx/fontforge` (heavy build dependency).
- Unifont is a pixel font; at 12 pt on the 1920x1200 / ~160 DPI eDP panel it
  does not look good anyway.

## Fix

### 1. Install proper monospace fonts (as root)

```bash
emerge --noreplace media-fonts/jetbrains-mono media-fonts/dejavu
fc-cache -f
```

DejaVu is installed as a general fallback (`DejaVu Sans Mono`).

Verify:

```bash
fc-match "JetBrains Mono"
fc-match "JetBrains Mono:bold"
fc-match "JetBrains Mono:italic"
```

Expected:

```text
JetBrainsMono-Regular.ttf: "JetBrains Mono" "Regular"
JetBrainsMono-Bold.ttf: "JetBrains Mono" "Bold"
JetBrainsMono-Italic.ttf: "JetBrains Mono" "Italic"
```

### 2. Update the Alacritty config

Back up the old config:

```bash
cp ~/.config/alacritty/alacritty.toml ~/.config/alacritty/alacritty.toml.unifont.bak
```

New `~/.config/alacritty/alacritty.toml`:

```toml
[selection]
save_to_clipboard = true

[font]
size = 11.0

[font.normal]
family = "JetBrains Mono"
style = "Regular"

[font.bold]
family = "JetBrains Mono"
style = "Bold"

[font.italic]
family = "JetBrains Mono"
style = "Italic"
```

Alacritty live-reloads its config; otherwise open a new window. Quick visual
check: JetBrains Mono has a dotted zero and clearly distinct `l` / `1` / `I`.

## Optional Tuning

- **Size:** adjust `font.size` (10–12 is reasonable on the E14 panel).
- **Subpixel rendering:** currently `10-sub-pixel-none.conf` is enabled. For
  sharper text on the LCD:

  ```bash
  eselect fontconfig disable 10-sub-pixel-none.conf
  eselect fontconfig enable 10-sub-pixel-rgb.conf
  ```

  Keep `11-lcdfilter-default.conf` enabled, then restart Alacritty.
- **Back to Unifont (if really wanted):**

  ```bash
  echo "media-fonts/unifont ttf utils" >> /etc/portage/package.use/unifont
  emerge media-fonts/unifont
  ```

## Persisting the Fix in the Image

The `emerge` above only lands in the running system's persistent overlay. To
keep it after rebuilding the squashfs image, add the fonts to
`config/var/lib/portage/world`:

```text
media-fonts/dejavu
media-fonts/jetbrains-mono
```

The Alacritty config lives in the user's home and is not part of the image.

## Lessons

- An unknown `family` in Alacritty does not produce an error; fontconfig just
  substitutes something, possibly a proportional font. Always check with
  `fc-match <family>`.
- Bitmap fonts are disabled by `70-no-bitmaps-except-emoji.conf` on this
  system.
