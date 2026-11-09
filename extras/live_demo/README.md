# Archived live demo

This folder is the archived pre-redesign prototype of the project, including a real loopback HTTP server that was used as a DDoS environment (the defense became serving capacity and the attack became a request flood).
It is not maintained against the current model in `DESIGN.md` and its agents, payoffs, and parameters differ from the main `security_game` package.
The modules use flat imports, so run them from inside this folder: `cd extras/live_demo`, then `python run_live.py --rounds 8` to play a few rounds of the old game against the live server, and `python validate.py` to compare the contest formula with measured server outages (writes `fig_validation.png`).
Both scripts need numpy, scipy, and matplotlib.
