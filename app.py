import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash, jsonify, redirect, url_for
import os
import time
from flask import Flask, render_template, jsonify, request
import pandas as pd
import module.wave_module as wave_module
import numpy as np
import time
from module.seabed import create_seabed_plot  # Ensure create_seabed_plot returns x, y, z arrays
import module.real_time as real
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import tkinter as tk
from module.seabed import create_seabed_plot  # Assumes this function generates x, y, z arrays for 3D plotting
import module.real_time as real
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from module.plotreal import plot_3d_curve_from_csv
import subprocess
import module.solver
from flask import session
from flask import Flask, render_template
import threading 
import module.filter
import module.frequencydomain_random as fr
import module.timedomain_random as tr 

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

DATA_FOLDER = "data"
bathy_file = None
start_time = time.time()
wave_angle = 90
time_period = 0

UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load seabed data with column names adjusted
seabed_data = pd.read_csv('run_data/depth.csv')
def start_plotter(z, y, x, t, alpha, r):
    # Ensure the plotting function doesn’t return anything
    # Run in a separate thread to avoid blocking Flask
    threading.Thread(target=module.solver.plotter, args=(z, y, x, t, alpha,r)).start()

def start_solver(z,depth,wave_height,time_period,wave_angle):
    # Ensure the plotting function doesn’t return anything
    # Run in a separate thread to avoid blocking Flask
    threading.Thread(target=module.solver.solve, args=(z,depth,wave_height,time_period,wave_angle)).start()


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_data = None
    if request.method == 'POST':
        try:
            # Fetching form data
            depths = request.form.getlist('depths[]')
            x_coords = request.form.getlist('x_coordinate[]')
            y_coords = request.form.getlist('y_coordinate[]')
            x_length = float(request.form.get('x_length', 1))
            y_length = float(request.form.get('y_length', 1))
            x_partition = int(request.form.get('x_partition', 100))
            y_partition = int(request.form.get('y_partition', 100))

            # Convert to floats, filtering out empty values
            depths = [float(d) for d in depths if d]
            x_coords = [float(x) for x in x_coords if x]
            y_coords = [float(y) for y in y_coords if y]

            print(depths)
            print(x_coords)
            print(y_coords)
            # Validate input lengths
            if not (len(depths) and len(x_coords) and len(y_coords)) or \
                    len(depths) != len(x_coords) or len(depths) != len(y_coords):
                flash("Please provide the same number of depths, x coordinates, and y coordinates.")
                return render_template('index.html', plot_data=plot_data)

            # Generate plot data as a dictionary
            plot_data = create_seabed_plot(depths, x_coords, y_coords, x_length, y_length, x_partition, y_partition)
        
        except ValueError:
            flash("Invalid input. Ensure numbers are correctly formatted.")
        except Exception as e:
            flash("An error occurred: " + str(e))

    return render_template('index.html', plot_data=plot_data)

@app.route('/list_csv_files', methods=['GET'])
def list_csv_files():
    try:
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        return jsonify(files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

   
@app.route('/real_depth', methods=['POST', 'GET'])
def real_depth():
    if request.method == 'POST':
        bathy_file = request.form.get('bathy_file')
        print("Selected bathy_file:", bathy_file)

    if not bathy_file:
        flash("Please select a CSV file.")
        return redirect(url_for('index'))

    file_path = os.path.join(DATA_FOLDER, bathy_file)
    plot_data = real.select_area_and_get_data(file_path)
    plot_3d_curve_from_csv()


    return render_template('index.html')


    # data = pd.read_csv(file_path)
    # x = data['X'].values
    # y = data['Y'].values
    # z = data['Depth'].values  # Assuming there is a Depth column

    # selected_points = []

    # def onselect(eclick, erelease):
    #     x_min, y_min = eclick.xdata, eclick.ydata
    #     x_max, y_max = erelease.xdata, erelease.ydata

    #     # Find points within the selected rectangle
    #     mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    #     selected_x = x[mask]
    #     selected_y = y[mask]

    #     if len(selected_x) > 0:
    #         # Adjust coordinates based on selection
    #         adjusted_x = selected_x - min(selected_x)
    #         adjusted_y = selected_y - min(selected_y)

    #         selected_points.extend(zip(adjusted_x, adjusted_y))

    #         # Close the plot after selection
    #         plt.close()

    # # Plotting the data in a Tkinter window
    # root = tk.Tk()
    # root.title("Select Area on 2D Plot")
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x, y, c=z, cmap='viridis')
    # plt.colorbar(scatter, label='Depth')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Select Area on 2D Plot')

    # # Initialize RectangleSelector
    # selector = RectangleSelector(ax, onselect, useblit=True, button=[1], spancoords='data')

    # plt.show(block=False)

    # # Start Tkinter main loop
    # root.mainloop()

    # # Process selected points after Tkinter window is closed
    # if selected_points:
    #     x_selected, y_selected = zip(*selected_points)
    #     z_selected = data[
    #         (data['X'].isin(x_selected)) & 
    #         (data['Y'].isin(y_selected))
    #     ]['Depth']

    #     msl = np.zeros(len(z_selected))  # Assuming you want a zero array for MSL
    #     print(z_selected)
    #     # Show selected data in a message box or process as needed
    #     return render_template('index.html',x=list(x_selected), y=list(y_selected), z=list(z_selected), msl=list(msl))
    
    # flash("No area selected. Please try again.")
    # return redirect(url_for('index'))
        
# @app.route('/real_depth', methods=['POST', 'GET'])
# def real_depth():
        
#     if not bathy_file:
#         bathy_file = request.form.get('bathy_file')
#         plot_data = None

#         # Fetch the selected CSV file path
#         file_path = os.path.join(DATA_FOLDER, bathy_file)

#         # Retrieve area limits, setting defaults if none provided
#         x_min = request.form.get('x_min', None)
#         x_max = request.form.get('x_max', None)
#         y_min = request.form.get('y_min', None)
#         y_max = request.form.get('y_max', None)

#         # Log the ranges for debugging
#         print(f"Ranges - x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

#         try:
#             # Generate plot data based on the selected area in the CSV file
#             plot_data = real.select_area_and_get_data(file_path, x_min, x_max, y_min, y_max)
            
#             # Generate both 3D and 2D plots
#             plot_images = create_3D_and_2D_plots(plot_data)
            
#             # Redirect to 'real_depth.html' with plot data
#             return render_template('real_depth.html', plot_data=plot_data, plot_images=plot_images)

#         except Exception as e:
#             flash(f"Error: {e}")
#             return redirect(url_for('index'))

#     # For GET requests, just render the page without plot data
#     return render_template('real_depth.html')

# def create_3D_and_2D_plots(plot_data):
#     # Assumes plot_data contains 'x', 'y', 'z' lists
#     x, y, z = plot_data['x'], plot_data['y'], plot_data['z']
#     fig_3d = plt.figure()
#     ax_3d = fig_3d.add_subplot(111, projection='3d')
#     ax_3d.plot_trisurf(x, y, z, cmap='viridis')
#     ax_3d.set_xlabel("X")
#     ax_3d.set_ylabel("Y")
#     ax_3d.set_zlabel("Depth")
#     fig_3d_path = os.path.join('static', 'plot_3d.png')
#     fig_3d.savefig(fig_3d_path)

#     fig_2d, ax_2d = plt.subplots()
#     ax_2d.tricontourf(x, y, z, cmap='viridis')
#     ax_2d.set_xlabel("X")
#     ax_2d.set_ylabel("Y")
#     fig_2d_path = os.path.join('static', 'plot_2d.png')
#     fig_2d.savefig(fig_2d_path)

#     return {'plot_3d': fig_3d_path, 'plot_2d': fig_2d_path}
@app.route('/range')
def range():
    seabed_data = pd.read_csv('run_data/depth.csv')
    x_min = seabed_data['X'].min()
    x_max = seabed_data['X'].max()
    y_min = seabed_data['Y'].min()
    y_max = seabed_data['Y'].max()
    return jsonify({'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})

@app.route('/data', methods=['POST'])
def data():
    seabed_data = pd.read_csv('run_data/depth.csv')
    area = request.json

    # Retrieve and convert area bounds to floats if they are provided, else use defaults from CSV
    x_min = float(area.get('x_min', seabed_data['X'].min()))
    x_max = float(area.get('x_max', seabed_data['X'].max()))
    y_min = float(area.get('y_min', seabed_data['Y'].min()))
    y_max = float(area.get('y_max', seabed_data['Y'].max()))

    # Filter the seabed data based on specified range
    seabed_filtered = seabed_data[
        (seabed_data['X'] >= x_min) & 
        (seabed_data['X'] <= x_max) & 
        (seabed_data['Y'] >= y_min) & 
        (seabed_data['Y'] <= y_max)
    ]
    wave_height = session.get('wave_height')
    time_period = session.get('time_period')
    wave_angle = session.get('wave_angle')

    # Generate wave data based on filtered seabed
    wave_data = wave_module.generate_wave(wave_angle,time_period, start_time)

    wave_filtered = wave_data[
        (wave_data['x'] >= x_min) & 
        (wave_data['x'] <= x_max) & 
        (wave_data['y'] >= y_min) & 
        (wave_data['y'] <= y_max)
    ]

    seabed_json = {
        'x': seabed_filtered['X'].tolist(),
        'y': seabed_filtered['Y'].tolist(),
        'depth': seabed_filtered['Depth'].tolist()
    }

    wave_json = {
        'x': wave_filtered['x'].tolist(),
        'y': wave_filtered['y'].tolist(),
        'wave': wave_filtered['wave_z'].tolist()
    }

    return jsonify({'seabed': seabed_json, 'wave': wave_json})

@app.route('/target')
def target_function():
    return render_template('tabs.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    # Capture form data
    wave_height = request.form.get('wave-height')
    time_period = request.form.get('time-period')
    wave_angle = request.form.get('wave-angle')

    session['wave_height'] = wave_height
    session['time_period'] = time_period
    session['wave_angle'] = wave_angle
    
    subprocess.run(['python', 'module/check.py','--T',str(time_period),'--H',str(wave_height),'--A',str(wave_angle)])

    subprocess.run(['python', 'module/details.py','--T',str(time_period),'--A',str(wave_angle)])
    # You can add any necessary processing with the captured inputs here
    # Optionally pass the inputs to the animate.html template for display or animation
    
    return render_template('animate.html')

@app.route('/Singlesolver', methods=['POST'])
def Singlesolver():
    wave_height = request.form.get('wave_height')
    time_period = request.form.get('time_period')
    wave_angle = request.form.get('incident_wave_angle')
    depth = request.form.get("depth")
    z = request.form.get("z_depth")
    print(time_period)

    start_solver(z,depth,wave_height,time_period,wave_angle)

    return "Plotting started. Check the separate window for the graph."

@app.route('/plotty', methods=['POST'])
def plot():
    x_values = request.form.get('x_values')
    y_values = request.form.get('y_values')
    z_values = request.form.get('z_values')
    s_radius = request.form.get('s_radius')
    wave_height = session.get('wave_height')
    time_period = session.get('time_period')
    wave_angle = session.get('wave_angle')

    start_plotter(z_values,y_values,x_values,time_period,wave_angle,s_radius)

    return "Plotting started. Check the separate window for the graph."

# Route to upload CSV files
@app.route('/upload', methods=['POST'])
def upload():
    if 'csv-upload' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['csv-upload']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # if not file.filename.endswith('.csv'):
    #     return jsonify({'error': 'Only CSV files are allowed'}), 400

    # Remove the .csv extension from the filename
    filename_without_extension = file.filename.rsplit('.csv', 1)[0]

    try:
        # Process the file content directly from memory
        module.filter.filter(file, filename_without_extension)
        return jsonify({'success': 'File uploaded and processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random',methods = ['POST'])
def random():

    if 'file' not in request.files:
        return "No file part", 400  # Return a status code for better handling

    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400  # Return a status code for better handling
    
    if file:
        # Read the CSV file into a pandas DataFrame
        file_path = os.path.join(DATA_FOLDER, file.filename)

        fr.plot_wave(file_path)
        tr.create_gui(file_path)
        
        return render_template('tabs.html')



if __name__ == '__main__':
    app.run(debug=True)
