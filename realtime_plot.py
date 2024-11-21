import re
import time
import matplotlib.pyplot as plt
import threading
from queue import Queue
import pyairview

# Store the RSSI values in a global list
rssi_values = []
plot_data_queue = Queue()

# Fixed frequency range for the plot
START_FREQ = 2399.0  # MHz
END_FREQ = 2485.0  # MHz
FREQ_STEP = 0.5  # MHz

# Fixed RSSI range for the plot (adjust this if your data has a wider range)
RSSI_MIN = -100  # dBm
RSSI_MAX = -40   # dBm

def parse_rssi_input(input_text):
    """
    Extracts RSSI readings from the input text using regex.
    """
    try:
        # Extract numbers inside square brackets after the "Received X RSSI level readings: "
        match = re.search(r"Received \d+ RSSI level readings: \[(.*)\]", input_text)
        if match:
            rssi_data = match.group(1)
            # Convert the extracted string into a list of integers
            readings = list(map(int, rssi_data.split(",")))
            return readings
        else:
            print("No valid RSSI data found in the input.")
            return []
    except Exception as e:
        print(f"Error parsing input: {e}")
        return []

def plot_rssi_spectrum():
    """
    Plots a spectrum analyzer style graph of RSSI values with frequency as the X-axis.
    This function continuously updates the plot from the queue.
    """
    fig, ax = plt.subplots()
    ax.set_title("RSSI Spectrum Analyzer")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("RSSI Level (dBm)")
    ax.grid(True)

    # Set the fixed limits for the frequency and RSSI axes
    ax.set_xlim(START_FREQ, END_FREQ)
    ax.set_ylim(RSSI_MIN, RSSI_MAX)

    while True:
        # Get new RSSI data from the queue
        if not plot_data_queue.empty():
            rssi_values = plot_data_queue.get()

            # Calculate the number of points based on the frequency range
            num_points = len(rssi_values)
            frequencies = [START_FREQ + i * FREQ_STEP for i in range(num_points)]

            # Plot the RSSI values against the frequencies
            ax.cla()  # Clear the axes
            ax.set_xlim(START_FREQ, END_FREQ)  # Reset frequency limits
            ax.set_ylim(RSSI_MIN, RSSI_MAX)    # Reset RSSI limits
            ax.plot(frequencies, rssi_values, marker='o', linestyle='-', color='b', alpha=0.7)
            ax.set_title("RSSI Spectrum Analyzer")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("RSSI Level (dBm)")
            ax.grid(True)
            plt.draw()
            plt.pause(0.1)  # Pause to update the plot

def scan_callback(rssi_list):
    """
    Callback function to handle received RSSI data.
    This function is called each time RSSI values are received from the Airview device.
    """
    print(f"Received {len(rssi_list)} RSSI level readings: {rssi_list}")
    
    # Put the new RSSI readings into the queue for plotting
    plot_data_queue.put(rssi_list)

def start_scan():
    """
    Starts the RSSI scan and processes the results continuously.
    """
    try:
        # Connect to the Airview device (assuming the device is connected and the port is correct)
        connected = pyairview.connect("/dev/ttyACM0")  # Replace with the correct port if necessary
        if not connected:
            print("Failed to connect to the device.")
            return
        
        print("Starting RSSI scan...")

        # Start scanning
        pyairview.start_scan(callback=scan_callback)
        
        # Start the plot update thread
        plot_thread = threading.Thread(target=plot_rssi_spectrum, daemon=True)
        plot_thread.start()

        # Continuously update the plot with new RSSI data
        while pyairview.is_scanning():
            time.sleep(0.5)  # Delay between scan updates

        # Stop scanning after a certain period or when finished
        pyairview.stop_scan()
        
    except KeyboardInterrupt:
        print("Scan interrupted by user.")
    except Exception as e:
        print(f"Error during scan: {e}")
    finally:
        # Ensure proper cleanup of the serial connection
        try:
            pyairview.disconnect()
            print("Disconnected from Airview device.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

        # Keep the plot open after scan completion
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the plot window open

if __name__ == "__main__":
    start_scan()


