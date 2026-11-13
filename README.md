# EyeComm – Python-only patient and caregiver prototype

This version keeps EyeComm in Python. It uses the latest supplied 2D gaze and blink code as the starting point, adds a queued text-to-speech service, and provides a local-network caregiver server/dashboard.

## Important scope

- Windows desktop prototype.
- Patient interface uses the existing OpenCV window.
- Eye tracking uses the existing MediaPipe Face Mesh approach.
- Gaze calibration remains the supplied five-target 2D calibration.
- Blink detection retains the supplied intentional-blink logic.
- Text-to-speech uses `pyttsx3`.
- Caregiver dashboard uses Tkinter.
- Patient/caregiver synchronisation uses a small FastAPI server and SQLite.
- No Flutter, Android, or iOS code is included.

This is a development prototype, not a clinically validated assistive device.

## 1. Python environment

Use Python 3.10 or 3.11 on Windows for the smoothest compatibility with the supplied MediaPipe Python implementation.

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 2. Run the local caregiver server

On the computer that will host the caregiver service:

```powershell
python run_server.py
```

The server listens on port 8765.

Check it in a browser:

`http://127.0.0.1:8765/api/health`

## 3. Run the caregiver dashboard

On the caregiver computer:

```powershell
python run_caregiver.py
```

The caregiver can:
- See incoming patient requests.
- Hear a sound for newly received urgent requests.
- Acknowledge a request.
- Mark a request resolved.
- Change the patient's button labels and spoken phrases.

## 4. Run the patient application

On the patient computer:

```powershell
python run_patient.py
```

The patient application:
- Uses the five-point gaze calibration.
- Uses intentional blink selection.
- Shows an 8-button communication board.
- Speaks composed messages when SPEAK / SEND is selected.
- Sends the composed message to the caregiver as a normal request.
- Sends HELP as an urgent request and speaks "I need help".

Keyboard controls:
- `Q` or `Esc`: quit.
- `K`: recalibrate.
- `S`: speak and send the current message.
- `U`: remove the last phrase.
- `X`: clear the message buffer.
- Calibration keys: `C`, `U`, `D`, `L`, `R` for centre, up, down, left, right.

## 5. Connect two Windows computers on the same network

Run `run_server.py` on the caregiver/server computer.

Find that computer's LAN IPv4 address (for example `192.168.1.25`). On the patient computer, edit:

`data/settings.json`

Change:

```json
"server_url": "http://127.0.0.1:8765"
```

to:

```json
"server_url": "http://192.168.1.25:8765"
```

Do the same on the caregiver computer if it is not using the default address.

Allow inbound TCP port 8765 through Windows Firewall on a trusted private network only.

Both computers must be able to reach the server. This initial implementation is for a trusted local network, not direct public-internet exposure.

## 6. Calibration profile compatibility

The older uploaded calibration file was version 2. The latest supplied 2D gaze code requires version 3, with centre, up, down, left and right measurements.

The old file is preserved in `legacy/calibration_profile_v2.json`. It is intentionally not copied into the active `data` folder. Run a fresh five-point calibration in this version.

## 7. What is preserved

The original supplied files are retained unchanged in `legacy/`.

The active vision module is based on the latest supplied `step3_calibratedgaze(1).py`. The blink detector is extracted from `step2_blinkdetector(1).py`. The patient application is based on `step5_dualsystem(1).py`.

The eye-metric formulas, landmark indices, calibration thresholds, gaze smoothing, directional navigation, and blink-duration rules have not intentionally been redesigned.

## 8. Current limitations / next development work

- The local server prototype has no user authentication or TLS. Do not expose it to the public internet.
- Patient identity is currently a local configured ID, not a verified login.
- The caregiver sound depends on the caregiver dashboard being open and the computer's audio being enabled.
- The patient currently polls for caregiver board changes every few seconds.
- The OpenCV patient interface is retained for minimal disruption. A more polished Python GUI (for example PySide6) can be a later UI-only migration.
- Add authentication, patient-caregiver pairing, access controls, encrypted transport, reconnect handling and stronger offline queuing before real-world deployment.
- Emergency requests are not a substitute for a medical emergency call system.

## 9. Suggested test order

1. Run `python -m compileall eyecomm`.
2. Run the server and check `/api/health`.
3. Open the caregiver dashboard.
4. Change a label and phrase, then save.
5. Start the patient app and confirm the board refreshes.
6. Complete fresh calibration.
7. Verify intentional blink unlocks the board.
8. Select WATER, then SPEAK / SEND; confirm local speech and caregiver request.
9. Select HELP; confirm urgent caregiver alert sound.
10. Acknowledge and resolve the request in the caregiver dashboard.
