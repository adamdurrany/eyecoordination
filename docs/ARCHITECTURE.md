# EyeComm Python architecture

```text
Patient PC
  OpenCV camera + MediaPipe Face Mesh
             |
             v
  Existing eye_metrics / 2D calibration
             |
             v
  Existing BlinkDetector + GazeFilter + Navigator
             |
             v
  Patient communication board
       |                |
       |                +--> SpeechService (pyttsx3)
       |
       +--> SyncClient (HTTP)
                    |
                    v
              FastAPI server
                    |
             SQLite + board.json
                    |
                    v
           Caregiver Tkinter dashboard
              |              |
          edit board      alert sound
              |              |
              +---- ACK / RESOLVE
```

## Data flow

- The caregiver edits tile labels and spoken phrases.
- The server validates the fixed action IDs and stores the board configuration.
- The patient app polls the board in a background thread.
- A normal composed message is sent when SPEAK / SEND is activated.
- HELP sends a request with `priority=urgent`.
- The caregiver dashboard polls the request list and sounds for newly observed urgent pending requests.
- Acknowledgement and resolution status are written to SQLite.

## Trust boundary

This first version is designed for a trusted local network. It does not implement account authentication, pairing secrets, TLS, or role-based access control. Those are required before use beyond a controlled prototype network.
