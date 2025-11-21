<h1 align="center">AI-Based Predictive Maintenance System for Automobiles</h1>

<p align="center">
  A backend system built with FastAPI that analyzes OBD-II data, detects anomalies, 
  and evaluates real-time engine health using machine learning.
</p>

<hr>

<h2>📌 Overview</h2>
<p>
The project focuses on using OBD-II sensor data to identify early signs of possible
issues in a vehicle. Instead of relying on fixed service intervals, the system
evaluates sensor behaviour, checks for abnormal readings, and flags potential
problems. It supports both real-time input and CSV uploads, making it useful for
testing, integration with hardware modules, or further application-level development.
</p>

<hr>

<h2>🚗 Key Features</h2>
<ul>
  <li><strong>Real-time analytics</strong> via FastAPI endpoint.</li>
  <li><strong>CSV upload</strong> support for batch evaluation of sensor logs.</li>
  <li><strong>Anomaly detection</strong> using Isolation Forest.</li>
  <li><strong>Engine-health assessment</strong> based on defined safe operating ranges.</li>
  <li><strong>SQLite database</strong> for storing incoming OBD-II data.</li>
  <li><strong>Graph data</strong> formatted for frontend visualisation.</li>
  <li><strong>Model retraining</strong> endpoint for updated behaviour as more data accumulates.</li>
</ul>

<hr>

<h2>🧩 Core Sensors Used</h2>
<p>The system evaluates the following OBD-II parameters:</p>
<ul>
  <li>Engine Coolant Temperature</li>
  <li>Engine RPM</li>
  <li>Short-Term Fuel Trim</li>
  <li>Long-Term Fuel Trim</li>
  <li>Oxygen Sensor Voltage (with oscillation check)</li>
  <li>Intake Air Temperature</li>
  <li>Mass Air Flow</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
├── obd_data/           # Stored CSVs + SQLite database
├── models/             # Isolation Forest + Scaler files
├── main.py             # FastAPI backend
└── README.html         # Project documentation
</pre>

<hr>

<h2>🧠 Machine Learning</h2>
<p>
The project uses <strong>Isolation Forest</strong> for detecting sensor behaviour
that deviates from normal patterns. A <strong>StandardScaler</strong> is used before
training and prediction. Both trained objects are saved locally and reused when the
backend restarts.
</p>

<hr>

<h2>🗄 Database</h2>
<p>
All data is stored in an SQLite database inside the <code>obd_data/</code> directory.
This includes timestamps, raw sensor values, and anomaly predictions.
</p>

<hr>

<h2>🛠 Requirements</h2>
<ul>
  <li>Python 3.10+</li>
  <li>FastAPI</li>
  <li>scikit-learn</li>
  <li>pandas</li>
  <li>sqlite3</li>
  <li>numpy</li>
</ul>

<hr>

<hr>

<h2>📌 Notes</h2>
<ul>
  <li>The project is not yet integrated with a physical OBD device. All evaluations are done using CSV uploads or real-time data.</li>
  <li>If any incoming sensor reading exceeds the defined safe thresholds, the system flags it as a warning and marks it as a possible maintenance requirement.</li>
  <li>The anomaly detection model becomes more reliable as more real driving data gets added to the database.</li>
  <li>Threshold values can be adjusted later based on specific vehicle behaviour or manufacturer guidelines.</li>
</ul>

<hr>

