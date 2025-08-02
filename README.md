# Pi Audio Summary

This project records audio segments, detects human language, and generates a daily summary of conversations. It can run either in a PC Docker environment or on a Raspberry Pi 4.

## PC Docker Usage

1. Build and start the container:
   ```bash
   docker compose up --build
   ```
   The service uses `Dockerfile.pc` and stores logs in the `data/` folder.

2. Audio is processed continuously. Run `python summarizer.py` inside the container to produce a daily narrative from `data/log.csv`.

## Raspberry Pi 4 Usage

1. Flash Raspberry Pi OS (64‑bit) onto a 64 GB or 128 GB microSD card.
2. Install Docker on the Pi and clone this repository.
3. Build and run using the Pi-specific compose file:
   ```bash
   docker compose -f docker-compose.pi.yaml up --build
   ```
   The build uses `Dockerfile.pi` which targets the ARM64 architecture.

4. To generate the daily wrap‑up, run:
   ```bash
   docker exec -it pi_listener python summarizer.py
   ```

The summaries attempt to capture who might be speaking, what was said, and any context implied in the recordings.
