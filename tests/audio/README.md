# tests/audio/

Place real audio files here for model self-testing via `/api/models/test`.

Recommended files (WAV, MP3, FLAC, or OGG):

| Filename              | Expected class       |
|-----------------------|----------------------|
| alarm.wav             | ALARM                |
| siren.wav             | SIREN                |
| bicycle.wav           | BICYCLE              |
| vehicle_horn.wav      | HORN                 |
| glass_break.wav       | GLASS_BREAK          |
| dog_bark.wav          | ANIMAL               |
| speech.wav            | SPEECH               |
| machinery.wav         | MACHINERY            |
| background_noise.wav  | BACKGROUND           |
| gunshot.wav           | GUNSHOT              |

The `/api/models/test` endpoint will automatically pick the first audio file
it finds in this directory and run both SonicSentinel and AST against it.

If no files are present, the endpoint falls back to a synthetic 440 Hz sine tone.

Sources for free test audio:
- https://freesound.org
- https://www.soundsnap.com
- https://pixabay.com/sound-effects/
