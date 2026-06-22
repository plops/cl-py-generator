# HP Z6 Audio Configuration Notes

This note captures what was learned while debugging audio on the HP Z6 Gentoo OpenRC image built from:

- `/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/Dockerfile`
- `/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/doc/install_on_hpz6.md`

## Summary

The machine has working ALSA playback and a visible ALSA capture device, but the microphone input itself appears effectively silent in the current setup.

What is working:

- ALSA playback opens and plays normally on the `Generic` card.
- ALSA capture opens on `hw:Generic,0` and `default`.
- The kernel exposes the expected codec and PCM nodes for the `HD-Audio Generic` / `ALC222 Analog` device.

What is not working:

- The captured waveform is effectively near zero amplitude.
- The microphone jack is not currently being asserted.
- PipeWire was not required for the final diagnosis, and the system can be treated as ALSA-only if desired.

## Observed Hardware

Relevant ALSA hardware reported by the system:

- `card 0`: `HDA NVidia`
- `card 1`: `HD-Audio Generic`
- Capture device:
  - `card 1, device 0`
  - `ALC222 Analog`

`/proc/asound/card1/codec#0` shows the Realtek codec as:

- `Codec: Realtek ALC222`
- `Subsystem Id: 0x103c8b24`

## ALSA Behavior

Playback:

- `speaker-test -D default -c 2 -r 44100` works.
- `aplay -D hw:Generic,0 /tmp/test.wav` works.

Capture:

- `arecord -D default -f S16_LE -r 44100 -c 2` produces a file.
- The recorded samples were essentially silence:
  - `max_abs` was only `179`
  - `rms` was about `2.9`

This means the input path is opening, but the selected mic signal is not useful.

## Mixer State

The `Generic` card exposes these useful controls:

- `Capture`
- `Front Mic`
- `Rear Mic`
- `Front Mic Boost`
- `Rear Mic Boost`
- `Loopback Mixing`
- `Auto-Mute Mode`

The current codec state shows:

- `Capture` is enabled and near maximum.
- `Front Mic Boost` and `Rear Mic Boost` are already at maximum.
- The mic jack detection bits were not asserted.

That suggests the problem is likely one of:

- wrong physical jack
- mic not plugged in
- broken microphone hardware
- wrong source path in the analog codec

## Practical ALSA-Only Baseline

If the goal is to keep the desktop ALSA-only, the simplest operational model is:

1. Do not launch PipeWire user services.
2. Keep ALSA as the only active desktop audio path.
3. Use `hw:Generic,0` or `sysdefault:CARD=Generic` for direct testing.

Useful test commands:

```bash
speaker-test -D default -c 2 -r 44100
aplay -D hw:Generic,0 /tmp/test.wav
arecord -D hw:Generic,0 -f S16_LE -r 44100 -c 2 /tmp/mic-test.wav
```

## Troubleshooting Notes

- If `arecord` returns a file but the file is almost silent, the capture path is open but the microphone source is probably wrong.
- If `amixer` shows the mic boosts and capture gain already high, further gain changes are unlikely to help.
- If the jack detect bits remain off, the hardware may simply not be seeing a plugged-in microphone.
- If playback works but capture remains silent across both front and rear mic paths, treat it as a hardware or jack-path issue first, not a Gentoo build issue.

## Conclusion

On this HP Z6, the audio stack itself is not the main problem anymore.

- ALSA playback works.
- ALSA capture opens.
- The microphone signal is effectively absent.

The current evidence points to either the wrong mic jack/path or a non-functional microphone input.
