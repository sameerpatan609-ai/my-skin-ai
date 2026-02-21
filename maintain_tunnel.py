import subprocess
import time
import os

def maintain_tunnel():
    print("Starting Self-Maintaining Tunnel...")
    # Use 127.0.0.1 for better tunnel stability
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:127.0.0.1:5000", "nokey@localhost.run"]
    
    while True:
        try:
            print(f"[{time.ctime()}] Connecting to tunnel...")
            with open("tunnel_log.txt", "w") as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=log, shell=False)
                
                # Check log every 5 seconds for the URL
                for _ in range(10):
                    time.sleep(2)
                    if os.path.exists("tunnel_log.txt"):
                        with open("tunnel_log.txt", "r") as r:
                            content = r.read()
                            if ".lhr.life" in content:
                                url = [line for line in content.split('\n') if ".lhr.life" in line][0]
                                print(f"ALIVE: {url}")
                                break
                
                process.wait() # Wait for it to die
                print(f"[{time.ctime()}] Tunnel dropped. Reconnecting in 5s...")
                time.sleep(5)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    maintain_tunnel()
