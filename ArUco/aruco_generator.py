
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# Define the ArUco dictionary
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Generate ArUco marker based on selected dictionary and ID
def generate_marker():
    # Get the selected dictionary and ID
    selected_dict = dictionary_var.get()
    marker_id = int(id_entry.get())

    # Check if the selected dictionary is valid
    if selected_dict not in ARUCO_DICT:
        messagebox.showerror("Error", "Invalid ArUco dictionary selected.")
        return

    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[selected_dict])

    # Generate the ArUco marker
    marker_size = 300
    marker = np.zeros((marker_size, marker_size, 1), dtype= "uint8")
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker, 1) 
    # According to OpenCV documentation, 6/4/23 method for Aruco marker generation changed from cv2.aruco.drawMarker to cv2.aruco.generateImageMarker().

    # Save the generated marker
    save_path = filedialog.asksaveasfilename(defaultextension=".png")
    if save_path:
        cv2.imwrite(save_path, marker)
        messagebox.showinfo("Success", "ArUco marker generated and saved successfully.")
    else:
        messagebox.showinfo("Cancelled", "ArUco marker generation cancelled.")

# Create the main window
window = tk.Tk()
window.title("ArUco Marker Generator")
window.geometry('600x600')

# Create the dictionary selection menu
dictionary_label = tk.Label(window, text="ArUco Dictionary:")
dictionary_label.pack()

dictionary_var = tk.StringVar(window, value="DICT_4X4_50")
dictionary_menu = tk.OptionMenu(window, dictionary_var, *ARUCO_DICT.keys())
dictionary_menu.pack()

# Create the ID input field
id_label = tk.Label(window, text="Marker ID:")
id_label.pack()

id_entry = tk.Entry(window)
id_entry.pack()

# Create the generate button
generate_button = tk.Button(window, text="Generate", command=generate_marker)
generate_button.pack()

# Start the main event loop
window.mainloop()
