import argparse
import csv
import time
from pathlib import Path

import serial
from serial.tools import list_ports


def pick_port(requested_port: str | None) -> str:
    if requested_port:
        return requested_port

    ports = [p.device for p in list_ports.comports()]
    if not ports:
        raise RuntimeError("No serial ports detected.")

    if len(ports) == 1:
        return ports[0]

    raise RuntimeError(f"Multiple ports found: {ports}. Use --port")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    port = pick_port(args.port)

    ser = serial.Serial(port, args.baud, timeout=1)

    out_path = Path(__file__).with_name("final_data.csv")
    out_file = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(out_file)

    # ✅ CSV HEADER (matches Arduino output)
    writer.writerow([
        "RangeOfMotion",
        "MotionSpeed",
        "PostureDuration",
        "Smoothness",
        "JerkRMS",
        "MeanVelocity",
        "MaxVelocity",
        "State",
        "LDH_Probability",
        "Risk"
    ])

    print(f"Logging from {port} → {out_path}")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            # ✅ Only process FEATURE lines
            if not line.startswith("FEATURE"):
                continue

            parts = line.split(",")

            # Remove "FEATURE"
            if parts[0] == "FEATURE":
                parts = parts[1:]

            if len(parts) != 10:
                continue  # safety check

            writer.writerow(parts)
            out_file.flush()

            print(parts)

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        ser.close()
        out_file.close()


if __name__ == "__main__":
    main()  