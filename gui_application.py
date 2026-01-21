import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from scipy.signal import welch     #calcul DSP
from scipy import fft
import cv2
from PIL import Image, ImageTk

# Import des modules du projet
from database_integration import AirQualityDatabase
from data_processing import DataProcessor
from correlation_analysis import CorrelationAnalyzer
from spectral_analysis import SpectralAnalyzer
from image_processing import ImageProcessor


class EnvironmentalDataGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Environmental Data Processing - Air quality and environmental metrics -")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)  #max réduire possible
        
        #variables
        self.db = AirQualityDatabase("air_quality.db")
        self.data = None
        self.COLUMN_MAP = {
            'CO': 'co_gt',
            'NO2': 'no2_gt',
            'Temperature': 'temperature',
            'Humidity': 'humidity'
}
        self.display_columns = list(self.COLUMN_MAP.keys())  # ['CO', 'NO2', 'Temperature', 'Humidity']


        self.current_image = None
        self.original_image = None
        self.image_processor = ImageProcessor()
        
        self.initialize_database()
    
        self.setup_style()
        
        self.create_menu()
        self.create_main_layout()
        
        self.load_data_from_db()
        

    
    
    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

    # Notebook (onglets)
        style.configure(
            "TNotebook",
            background="#F4F7F4",
            borderwidth=0
    )

# --- Onglets : même taille, pas de rétrécissement, pas de pointillés ---
        style.configure(
            "TNotebook.Tab",
            background="#DDEAE2",
            foreground="#2F3E36",
            padding=(18, 8),      # identique pour tous
            font=("Helvetica", 9, "bold"),
            borderwidth=0,
            relief="flat"
)

        style.map(
            "TNotebook.Tab",
            background=[("selected", "#A8C3B1")],
            foreground=[("selected", "#1F2A24")],
            padding=[("selected", (18, 8))],   # identique pour onglet actif
            relief=[("selected", "flat")],
            focuscolor=[("focus", "")],        # désactive le focus pointillé
            bordercolor=[("focus", "")],
            lightcolor=[("focus", "")],
            darkcolor=[("focus", "")]
)

    

    # Frames
        style.configure(
            "TFrame",
            background="#FAFBFA"
    )

        style.configure(
            "TLabelframe",
            background="#FAFBFA",
            foreground="#2F3E36",
            font=("Helvetica", 10, "bold")
    )

        style.configure(
            "TLabelframe.Label",
            background="#FAFBFA",
            foreground="#2F3E36"
    )

    # Labels
        style.configure(
            "TLabel",
            background="#FAFBFA",
            foreground="#2F3E36",
            font=("Helvetica", 10)
    )

        style.configure(
            "Header.TLabel",
            font=("Helvetica", 12, "bold"),
            foreground="#2F3E36",
            background="#FAFBFA"
    )

    # Boutons
        style.configure(
            "TButton",
            background="#A8C3B1",
            foreground="#1F2A24",
            font=("Helvetica", 9, "bold"),
            padding=6,
            borderwidth=0
    )

        style.map(
            "TButton",
            background=[
                ("active", "#C6DDD1"),
                ("pressed", "#7FA892")
        ]
    )

    # Scrollbar
        style.configure(
            "Vertical.TScrollbar",
            background="#DDEAE2",
            troughcolor="#F4F7F4"
    )

    
    def create_menu(self):
        """Crée la barre de menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        #menu file
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="CSV loading", command=self.load_csv_dialog)
        file_menu.add_command(label="Export data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        
        #menu Base de données
        db_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Database", menu=db_menu)
        db_menu.add_command(label="Reload data", command=self.load_data_from_db)
        db_menu.add_command(label="Statistics", command=self.show_db_stats)
        db_menu.add_command(label="Clean data", command=self.clean_data)
        
        # Menu help
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        #frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        #onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.configure(takefocus=0)  # empêche le focus, plus de pointillés


        self.create_data_tab()
        self.create_filter_tab()
        self.create_correlation_tab()
        self.create_spectral_tab()
        self.create_image_tab()
 
        
        #zone de log en bas
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding=5)
        log_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=4, font=('Consolas', 9))
        self.log_text.pack(fill=tk.X)
    
    #ONGLET 1: DONNÉES 
    
    def create_data_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Data")
        
        left_frame = ttk.LabelFrame(tab, text="Data Management", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        
        ttk.Label(left_frame, text="━━ Import ━━", font=('Helvetica', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(left_frame, text="Import CSV", command=self.load_csv_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Load Data from Database", command=self.load_data_from_db).pack(fill=tk.X, pady=2)
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left_frame, text="━━ Export ━━", font=('Helvetica', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(left_frame, text="Export Data to CSV", command=self.export_data).pack(fill=tk.X, pady=2)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left_frame, text="━━ Database ━━", font=('Helvetica', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(left_frame, text="Clear database", command=self.clear_database).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Store Data to Database", command=self.save_to_db).pack(fill=tk.X, pady=2)
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(left_frame, text="Statistics:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.stats_label = ttk.Label(left_frame, text="No data", justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W, pady=5)
        
        #tableau de données
        right_frame = ttk.LabelFrame(tab, text="Data overview", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        columns = ('ID', 'Date', 'Time', 'CO', 'NO2','Temperature', 'Humidity')
        self.data_tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        #scrollbars
        vsb = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        hsb = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
    
    #ONGLET 2: filtrage
    
    def create_filter_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Filtering")
        
        left_frame = ttk.LabelFrame(tab, text="Filter Settings", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        #variables
        ttk.Label(left_frame, text="Variable:").pack(anchor=tk.W)
        self.filter_var = ttk.Combobox(left_frame, values=self.display_columns, state="readonly")
        self.filter_var.set(self.display_columns[0])

        self.filter_var.pack(fill=tk.X, pady=5)
        
        #type filtre
        ttk.Label(left_frame, text="Filter Type:").pack(anchor=tk.W, pady=(10, 0))
        self.filter_type = ttk.Combobox(left_frame, values=['Moving Average', 'Threshold Filter'])
        self.filter_type.set('Moving Average')
        self.filter_type.pack(fill=tk.X, pady=5)
        
        #Paramètres
        ttk.Label(left_frame, text="Window Size:").pack(anchor=tk.W, pady=(10, 0))
        self.window_size = ttk.Scale(left_frame, from_=3, to=50, orient=tk.HORIZONTAL)
        self.window_size.set(10)
        self.window_size.pack(fill=tk.X, pady=5)
        self.window_label = ttk.Label(left_frame, text="10")
        self.window_label.pack()
        self.window_size.configure(command=lambda v: self.window_label.configure(text=f"{int(float(v))}"))
        
        #seuil
        ttk.Label(left_frame, text="Min Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.threshold_min = ttk.Entry(left_frame)
        self.threshold_min.insert(0, "0")
        self.threshold_min.pack(fill=tk.X, pady=5)
        
        ttk.Label(left_frame, text="Max Threshold:").pack(anchor=tk.W)
        self.threshold_max = ttk.Entry(left_frame)
        self.threshold_max.insert(0, "50")
        self.threshold_max.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Apply Filter", command=self.apply_filter).pack(fill=tk.X, pady=20)
        ttk.Button(left_frame, text="Reset", command=self.reset_filter).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Save Filtered Data to DB", command=self.save_filtered_data).pack(fill=tk.X, pady=5)

        right_frame = ttk.LabelFrame(tab, text="Filter Visualization", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
       
        # Frame pour le tableau filtré
        filtered_frame = ttk.LabelFrame(tab, text="Filtered Data", padding=10)
        filtered_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0))

        # Créer le Treeview
        self.filtered_tree = ttk.Treeview(filtered_frame, columns=('Index', 'Value'), show='headings', height=20)
        self.filtered_tree.heading('Index', text='Index')
        self.filtered_tree.heading('Value', text='Value')
        self.filtered_tree.column('Index', width=60, anchor='center', stretch=False)
        self.filtered_tree.column('Value', width=80, anchor='center', stretch=False)

        # Scrollbars
        vsb_f = ttk.Scrollbar(filtered_frame, orient=tk.VERTICAL, command=self.filtered_tree.yview)
        hsb_f = ttk.Scrollbar(filtered_frame, orient=tk.HORIZONTAL, command=self.filtered_tree.xview)
        self.filtered_tree.configure(yscrollcommand=vsb_f.set, xscrollcommand=hsb_f.set)

        self.filtered_tree.grid(row=0, column=0, sticky='nsew')
        vsb_f.grid(row=0, column=1, sticky='ns')
        hsb_f.grid(row=1, column=0, sticky='ew')

        filtered_frame.grid_rowconfigure(0, weight=1)
        filtered_frame.grid_columnconfigure(0, weight=1)

        
        self.filter_fig = Figure(figsize=(8, 6), dpi=100)
        self.filter_canvas = FigureCanvasTkAgg(self.filter_fig, right_frame)
        self.filter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.filter_canvas, toolbar_frame)
    
    #ONGLET 3: CORRÉLATION
    
    def create_correlation_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Correlations")
        
        left_frame = ttk.LabelFrame(tab, text="Settings", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(left_frame, text="Method:").pack(anchor=tk.W)
        self.corr_method = ttk.Combobox(left_frame, values=['pearson', 'spearman'])
        self.corr_method.set('pearson')
        self.corr_method.pack(fill=tk.X, pady=5)
        
        #variables pour scatter
        ttk.Label(left_frame, text="Variable X:").pack(anchor=tk.W, pady=(10, 0))
        self.corr_var_x = ttk.Combobox(left_frame, values=self.display_columns)
        self.corr_var_x.set(self.display_columns[0])
        self.corr_var_x.pack(fill=tk.X, pady=5)
        
        ttk.Label(left_frame, text="Variable Y:").pack(anchor=tk.W)
        self.corr_var_y = ttk.Combobox(left_frame, values=self.display_columns)
        self.corr_var_y.set(self.display_columns[1])
        self.corr_var_y.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Show Heatmap", command=self.show_correlation_heatmap).pack(fill=tk.X, pady=20)
        ttk.Button(left_frame, text="Show Scatter", command=self.show_scatter_plot).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Save to Database", command=self.save_correlations).pack(fill=tk.X, pady=5)
        
        # Résultats
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Results:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.corr_result_label = ttk.Label(left_frame, text="", justify=tk.LEFT, wraplength=200)
        self.corr_result_label.pack(anchor=tk.W, pady=5)
        
        right_frame = ttk.LabelFrame(tab, text="Graph", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.corr_fig = Figure(figsize=(8, 6), dpi=100)
        self.corr_canvas = FigureCanvasTkAgg(self.corr_fig, right_frame)
        self.corr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.corr_canvas, toolbar_frame)
    
    #ONGLET 4: ANALYSE SPECTRALE
    
    def create_spectral_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Spectral (FFT)")
        
        left_frame = ttk.LabelFrame(tab, text="FFT Settings", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Variable
        ttk.Label(left_frame, text="Variable:").pack(anchor=tk.W)
        self.spectral_var = ttk.Combobox(left_frame, values=self.display_columns)
        self.spectral_var.set(self.display_columns[0])
        self.spectral_var.pack(fill=tk.X, pady=5)
        
        #type de visualisation
        ttk.Label(left_frame, text="Type:").pack(anchor=tk.W, pady=(10, 0))
        self.spectral_type = ttk.Combobox(left_frame, values=['FFT', 'Power Spectrum(Welch)', 'periodogram'])
        self.spectral_type.set('Power Spectrum (Welch)')
        self.spectral_type.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Analyze", command=self.run_spectral_analysis).pack(fill=tk.X, pady=20)
        ttk.Button(left_frame, text="Save to Database", command=self.save_spectral_results).pack(fill=tk.X, pady=5)
        
        #résults
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Dominant Frequencies:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        self.spectral_results = scrolledtext.ScrolledText(left_frame, height=10, width=25, font=('Consolas', 9))
        self.spectral_results.pack(fill=tk.X, pady=5)
        
        # Frame droite - Graphique
        right_frame = ttk.LabelFrame(tab, text="Spectrum", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.spectral_fig = Figure(figsize=(8, 6), dpi=100)
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_fig, right_frame)
        self.spectral_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.spectral_canvas, toolbar_frame)
    
    #ONGLET 5: TRAITEMENT D'IMAGES
    
    def create_image_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Image Processing")

        left_container = ttk.LabelFrame(tab, text="Processing", padding=5)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        left_frame = self.create_scrollable_frame(left_container)

        
        ttk.Button(left_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Load Image from DB", command=self.load_image_from_db).pack(fill=tk.X, pady=5)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(left_frame, text="Processing:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        ttk.Button(left_frame, text="Grayscale Conversion", command=lambda: self.apply_image_processing('grayscale')).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Gaussian Blur", command=lambda: self.apply_image_processing('blur')).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Canny edge detection", command=lambda: self.apply_image_processing('canny')).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Otsu Thresholding", command=lambda: self.apply_image_processing('otsu')).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Adaptive Thresholding", command=lambda: self.apply_image_processing('adaptive')).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Sobel", command=lambda: self.apply_image_processing('sobel')).pack(fill=tk.X, pady=3)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Paramètre flou
       # Paramètre flou (KERNEL IMPAIR UNIQUEMENT)
        ttk.Label(left_frame, text="Blur Kernel:").pack(anchor=tk.W)

        self.blur_kernel = ttk.Scale(
           left_frame,
           from_=3,
           to=21,
           orient=tk.HORIZONTAL,
           command=self.update_blur_kernel
)
        self.blur_kernel.set(5)
        self.current_blur_kernel = 5

        self.blur_kernel.pack(fill=tk.X, pady=5)

        self.blur_kernel_label = ttk.Label(left_frame, text="5")
        self.blur_kernel_label.pack(anchor=tk.W)

        # Seuils Canny
        ttk.Label(left_frame, text="Canny Threshold 1:").pack(anchor=tk.W)
        self.canny_thresh1 = ttk.Scale(left_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.canny_thresh1.set(50)
        self.canny_thresh1.pack(fill=tk.X, pady=2)
        
        self.canny_thresh1_label = ttk.Label(left_frame, text=f"{int(self.canny_thresh1.get())}")
        self.canny_thresh1_label.pack(anchor=tk.W)
        self.canny_thresh1.configure(command=lambda v: self.canny_thresh1_label.configure(text=f"{int(float(v))}"))
        
        ttk.Label(left_frame, text="Canny Threshold 2:").pack(anchor=tk.W)
        self.canny_thresh2 = ttk.Scale(left_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.canny_thresh2.set(150)
        self.canny_thresh2.pack(fill=tk.X, pady=2)
        
        self.canny_thresh2_label = ttk.Label(left_frame, text=f"{int(self.canny_thresh2.get())}")
        self.canny_thresh2_label.pack(anchor=tk.W)
        self.canny_thresh2.configure(command=lambda v: self.canny_thresh2_label.configure(text=f"{int(float(v))}"))
        
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Button(left_frame, text="Reset", command=self.reset_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Save", command=self.save_processed_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Store Metadata", command=self.store_processed_image_metadata).pack(fill=tk.X, pady=5)
        
        #à droite
        right_frame = ttk.Frame(tab)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        orig_frame = ttk.LabelFrame(right_frame, text="Original Image", padding=5)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.orig_image_label = ttk.Label(orig_frame, text="No Image Loaded")
        self.orig_image_label.pack(fill=tk.BOTH, expand=True)

        proc_frame = ttk.LabelFrame(right_frame, text="Processed Image", padding=5)
        proc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.proc_image_label = ttk.Label(proc_frame, text="No Processing")
        self.proc_image_label.pack(fill=tk.BOTH, expand=True)
    
   
    
    #FONCTIONS UTILITAIRES
    
    def create_scrollable_frame(self, parent, width=200):
        canvas = tk.Canvas(parent, width=width, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)

        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        return scrollable_frame

    
    def log(self, message):
        #ajoute un message au journal
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def update_stats(self):
  #mise à jour des stats affichées
        if self.data is not None and len(self.data) > 0 and 'temperature' in self.data.columns:
            stats = f"""

Temperature:
  Moy: {self.data['temperature'].mean():.1f}°C
  Min: {self.data['temperature'].min():.1f}°C
  Max: {self.data['temperature'].max():.1f}°C

Humidity:
  Moy: {self.data['humidity'].mean():.1f}%"""
            self.stats_label.configure(text=stats)
        else:
            self.stats_label.configure(text="No Data\n\nLoad a CSV file\nto get started.")
    
    def update_data_tree(self):
        # Effacer les anciennes données
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.data is not None and len(self.data) > 0:
            # Afficher tous les enregistrements
            for idx, row in self.data.iterrows():
                values = (
                    row.get('id', idx),
                    row.get('date', ''),
                    row.get('time', ''),
                    f"{row.get('co_gt', 0):.2f}" if pd.notna(row.get('co_gt')) else '',
                    f"{row.get('no2_gt', 0):.2f}" if pd.notna(row.get('no2_gt')) else '',
                    f"{row.get('temperature', 0):.1f}" if pd.notna(row.get('temperature')) else '',
                    f"{row.get('humidity', 0):.1f}" if pd.notna(row.get('humidity')) else ''
                )
                self.data_tree.insert('', tk.END, values=values)
    
    #FONCTIONS DONNÉES 
    
    def initialize_database(self):
        try:
            self.db.connect()
            self.db.create_tables()
            self.db.disconnect()
        except Exception as e:
            print(f"Error initializing the database: {e}")
    
    def load_data_from_db(self):
        try:
            self.db.connect()
            self.db.create_tables()
            self.data = self.db.get_data_as_dataframe()

            if self.data is None or self.data.empty:
                self.data = pd.DataFrame()
                self.log("Database is empty. Please load a CSV file.")
            else:
                self.log(f"Data Loaded: {len(self.data)} Records")

        # Mise à jour de l'affichage
            self.update_stats()
            self.update_data_tree()
            print("Columns in DataFrame:", self.data.columns.tolist())
            print(self.data.head())

        except Exception as e:
            self.data = pd.DataFrame()
            self.log(f"Error loading database: {e}")
            self.update_stats()
            self.update_data_tree()
        finally:
            self.db.disconnect()
    
    def load_csv_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                #assurer que les tables existnt
                self.db.connect()
                self.db.create_tables()
                self.db.disconnect()
                
                #charger et nettoyer les données
                processor = DataProcessor()
                processor.load_data_from_csv(file_path)
                processor.clean_data()
                processor.store_cleaned_data()
                
                self.load_data_from_db()
                self.log(f"CSV Loaded and Stored: {file_path}")
                messagebox.showinfo("Success", "Data imported successfully!")
                
            except Exception as e:
                self.log(f"CSV Error: {str(e)}")
                messagebox.showerror("Error", f"Unable to load CSV: {str(e)}")
    
    def save_to_db(self):
      #sauvegarde les données dans la base
        if self.data is not None:
            try:
                processor = DataProcessor()
                processor.cleaned_data = self.data
                processor.store_cleaned_data()
                self.log("Data saved to the database")
                messagebox.showinfo("Success", "Data Saved!")
            except Exception as e:
                self.log(f"Error: {str(e)}")
    
    def clear_database(self):
        if messagebox.askyesno("Confirmation", "Are you sure you want to clear the database?"):
            try:
                self.db.connect()
                self.db.cursor.execute("DELETE FROM air_quality_measurements")
                #Réinitialiser le compteur d'auto-increment 
                self.db.cursor.execute("DELETE FROM sqlite_sequence WHERE name='air_quality_measurements'")
                self.db.connection.commit()
                self.db.disconnect()
                
                self.data = pd.DataFrame()
                self.update_stats()
                self.update_data_tree()
                self.log("Database cleared")
                messagebox.showinfo("Success", "The database has been cleared.")
            except Exception as e:
                self.log(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Unable to clear the database: {str(e)}")
    
    def export_data(self):
        #Exporte les données en CSV
        if self.data is None or len(self.data) == 0:
            messagebox.showwarning("Warning", "No data to export.\nPlease import a CSV file first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            self.data.to_csv(file_path, index=False)
            self.log(f"Data Exported: {file_path}")
            messagebox.showinfo("Success", f"Data exported to:\n{file_path}")
    
    def show_db_stats(self):
        #Afficher les statisde la base de données
        try:
            self.db.connect()
            stats = self.db.get_statistics()
            self.db.disconnect()
            
            msg = f"Total Records: {stats['total_records']}\n\n"
            for col, values in stats.items():
                if col != 'total_records' and isinstance(values, dict):
                    msg += f"{col}:\n"
                    msg += f"  Average: {values['moyenne']}\n"
                    msg += f"  Min: {values['min']}\n"
                    msg += f"  Max: {values['max']}\n\n"
            
            messagebox.showinfo("Database Statistics", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def clean_data(self):
        #Nettoie les données
        if self.data is not None:
            try:
                processor = DataProcessor()
                processor.data = self.data
                processor.clean_data()
                self.data = processor.cleaned_data
                self.update_stats()
                self.update_data_tree()
                self.log("Data Cleaned")
            except Exception as e:
                self.log(f"Error: {str(e)}")
    
    #FONCTIONS FILTRAGE 
    
    def apply_filter(self):
        #Applique le filtre sélectionné
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        var = self.filter_var.get()
        filter_type = self.filter_type.get()
        window = int(float(self.window_size.get()))
        
        self.filter_fig.clear()
        ax1 = self.filter_fig.add_subplot(211)
        ax2 = self.filter_fig.add_subplot(212)
        
        var_selected = self.filter_var.get()         
        df_column = self.COLUMN_MAP[var_selected]  
        original = self.data[df_column].dropna().values[:1000]

        
        #Appliquer le filtre
        if filter_type == 'Moving Average':
            filtered = pd.Series(original).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            title = f"Moving Average (window={window})"
        elif filter_type == 'Threshold Filter':
            min_val = float(self.threshold_min.get())
            max_val = float(self.threshold_max.get())
            filtered = np.clip(original, min_val, max_val)
            title = f"Thresholding [{min_val}, {max_val}]"
        else:  # Outliers
            q1, q3 = np.percentile(original, [25, 75])
            iqr = q3 - q1
            filtered = np.clip(original, q1 - 1.5*iqr, q3 + 1.5*iqr)
            title = "Remove Outliers (IQR)"
            # Mettre à jour le tableau filtered_tree
        for item in self.filtered_tree.get_children():
            self.filtered_tree.delete(item)

        for i, val in enumerate(filtered):
            self.filtered_tree.insert('', tk.END, values=(i, f"{val:.2f}"))

        
        # Graphiques
        ax1.plot(original, 'b-', linewidth=0.5, alpha=0.7, label='Original')
        ax1.set_title(f'Original Signal: {var}', fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(original, 'b-', linewidth=0.3, alpha=0.3, label='Original')
        ax2.plot(filtered, 'r-', linewidth=1, label=title)
        ax2.set_title(f'Filtered Signal: {title}', fontweight='bold')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.filter_fig.tight_layout()
        self.filter_canvas.draw()
        
        self.log(f"Filter Applied: {filter_type} to {var}")
    
    def reset_filter(self):
        #Réinitialise le graphique de filtrage
        self.filter_fig.clear()
        self.filter_canvas.draw()
        self.log("Filtering Reset")
        
    def save_filtered_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded to save")
            return
    
        var_selected = self.filter_var.get()
        filter_type = self.filter_type.get()
        window = int(float(self.window_size.get()))
    
        df_column = self.COLUMN_MAP[var_selected]
        original = self.data[df_column].dropna().values[:1000]
    
    # Appliquer le filtre
        if filter_type == 'Moving Average':
            filtered = pd.Series(original).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        elif filter_type == 'Threshold Filter':
            min_val = float(self.threshold_min.get())
            max_val = float(self.threshold_max.get())
            filtered = np.clip(original, min_val, max_val)
        else:
            q1, q3 = np.percentile(original, [25, 75])
            iqr = q3 - q1
            filtered = np.clip(original, q1 - 1.5*iqr, q3 + 1.5*iqr)
    
    # Mettre à jour le DataFrame
        self.data.loc[:len(filtered)-1, df_column] = filtered
    
    # Sauvegarder dans la base
        try:
            processor = DataProcessor()
            processor.cleaned_data = self.data
            processor.store_cleaned_data()
            self.log(f"Filtered data saved to database ({var_selected})")
            messagebox.showinfo("Success", f"Filtered data for {var_selected} saved to database.")
        except Exception as e:
            self.log(f"Error saving filtered data: {e}")
            messagebox.showerror("Error", f"Unable to save filtered data: {e}")

    
    #FONCTIONS CORRÉLATIONS
    
    def show_correlation_heatmap(self):
        # la heatmap des corrélations
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        method = self.corr_method.get()
        
        cols = [self.COLUMN_MAP[col] for col in self.display_columns if self.COLUMN_MAP[col] in self.data.columns]
        
        corr_matrix = self.data[cols].corr(method=method)
        inverse_map = {v: k for k, v in self.COLUMN_MAP.items()}
        corr_matrix = corr_matrix.rename(index=inverse_map, columns=inverse_map)
        self.corr_fig.clear()
        ax = self.corr_fig.add_subplot(111)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontweight='bold')
        
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()
        
        self.log(f"Correlation Heatmap Generated ({method})")
    
    def show_scatter_plot(self):
        #Affiche un scatter plot entre deux variables
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        var_x = self.COLUMN_MAP[self.corr_var_x.get()]
        var_y = self.COLUMN_MAP[self.corr_var_y.get()]

        
        self.corr_fig.clear()
        ax = self.corr_fig.add_subplot(111)
        
        ax.scatter(self.data[var_x], self.data[var_y], alpha=0.3, s=10, c='steelblue')
        
        #régression
        mask = ~(self.data[var_x].isna() | self.data[var_y].isna())
        z = np.polyfit(self.data.loc[mask, var_x], self.data.loc[mask, var_y], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.data[var_x].min(), self.data[var_x].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Regression')
        
        corr = self.data[var_x].corr(self.data[var_y])
        
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        ax.set_title(f'{var_x} vs {var_y} (r = {corr:.3f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()
        
        self.corr_result_label.configure(text=f"Correlation:\nr = {corr:.4f}")
        self.log(f"Scatter plot: {var_x} vs {var_y} (r={corr:.3f})")
    
    def save_correlations(self):
        #sauvegarde les corrélations dans la base de données
        try:
            analyzer = CorrelationAnalyzer()
            analyzer.load_data()
            analyzer.store_correlation_results(method=self.corr_method.get())
            self.log("Correlations Saved to Database")
            messagebox.showinfo("Success", "Correlations Saved!")
        except Exception as e:
            self.log(f"Error: {str(e)}")
    
    #FONCTIONS ANALYSE SPECTRALE
    
    def run_spectral_analysis(self):
        #Exécute l'analyse spectrale
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        var = self.spectral_var.get()
        spectral_type = self.spectral_type.get()
        
        var_selected = self.spectral_var.get()
        var_fft = self.COLUMN_MAP[var_selected]
        signal = self.data[var_fft].dropna().values

        
        self.spectral_fig.clear()
        
        if spectral_type == 'FFT':
            ax1 = self.spectral_fig.add_subplot(211)
            ax2 = self.spectral_fig.add_subplot(212)
            
            #signal
            ax1.plot(signal[:500], 'b-', linewidth=0.5)
            ax1.set_title(f'Signal: {var}', fontweight='bold')
            ax1.set_xlabel('Time (h)')
            ax1.grid(True, alpha=0.3)
            
            #FFT
            n = len(signal)
            fft_result = fft.fft(signal)
            frequencies = fft.fftfreq(n, d=1.0)
            amplitudes = np.abs(fft_result) * 2 / n
            
            mask = frequencies > 0
            ax2.plot(frequencies[mask][:n//4], amplitudes[mask][:n//4], 'r-', linewidth=0.5)
            ax2.set_title('FFT Spectrum', fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
        else:  #Welch ou Periodogramme
            ax = self.spectral_fig.add_subplot(111)
            
            frequencies, power = welch(signal, fs=1.0, nperseg=min(256, len(signal)//4))
            
            ax.semilogy(frequencies, power, 'b-', linewidth=1)
            ax.axvline(x=1/24, color='r', linestyle='--', alpha=0.7, label='24h')
            ax.axvline(x=1/12, color='g', linestyle='--', alpha=0.7, label='12h')
            ax.set_title(f'Power Spectrum: {var}', fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Spectral Density')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
        
        self.spectral_fig.tight_layout()
        self.spectral_canvas.draw()
        
        #frréqu dominantes
        if 'frequencies' in dir() and 'amplitudes' in dir():
            peak_idx = np.argsort(amplitudes[mask])[-5:][::-1]
            results = "Dominant Frequencies:\n"
            for idx in peak_idx:
                f = frequencies[mask][idx]
                if f > 0:
                    period = 1/f
                    results += f"• {f:.5f} Hz ({period:.1f}h)\n"
            self.spectral_results.delete(1.0, tk.END)
            self.spectral_results.insert(tk.END, results)
        
        self.log(f"Spectral Analysis: {var}")
    
    def save_spectral_results(self):
        #Sauvegarde les résultats spectraux
        try:
            analyzer = SpectralAnalyzer()
            analyzer.load_data()
            analyzer.store_spectral_results(self.spectral_var.get())
            self.log("Spectral Results Saved")
            messagebox.showinfo("Success", "Results Saved!")
        except Exception as e:
            self.log(f"Error: {str(e)}")
    
    #FONCTIONS IMAGES
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")]
        )
        
        if file_path:
            try:
                self.image_processor.load_image(file_path)
                self.original_image = self.image_processor.original_image.copy()
                self.current_image = self.original_image.copy()
                
                self.display_images()
                self.log(f"Image Loaded: {file_path}")
            except Exception as e:
                self.log(f"Error: {str(e)}")
    
    def display_images(self):
        #Affiche les images originale et traité
        if self.original_image is not None:
            #original img
            img_orig = self.resize_image_for_display(self.original_image)
            self.orig_photo = ImageTk.PhotoImage(img_orig)
            self.orig_image_label.configure(image=self.orig_photo)
        
        if self.current_image is not None:
            #image traitée
            img_proc = self.resize_image_for_display(self.current_image)
            self.proc_photo = ImageTk.PhotoImage(img_proc)
            self.proc_image_label.configure(image=self.proc_photo)
    
    def resize_image_for_display(self, img, max_size=400):
        #Redimensionne image pour l'affichage
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        h, w = img_rgb.shape[:2]
        scale = min(max_size/w, max_size/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        return Image.fromarray(img_resized)
    
    def apply_image_processing(self, operation):
        #Applique filtres pr image
        if self.original_image is None:
            messagebox.showwarning("Warning", "No Image Loaded")
            return
        
        try:
            self.image_processor.image = self.current_image.copy()
            
            if operation == 'grayscale':
                self.current_image = self.image_processor.convert_to_grayscale()
            elif operation == 'blur':
                kernel = self.current_blur_kernel
                self.current_image = self.image_processor.apply_gaussian_blur(kernel_size=kernel)
            elif operation == 'canny':
                t1 = int(float(self.canny_thresh1.get()))
                t2 = int(float(self.canny_thresh2.get()))
                self.current_image = self.image_processor.detect_edges_canny(threshold1=t1, threshold2=t2)
            elif operation == 'otsu':
                self.current_image = self.image_processor.apply_threshold(method='otsu')
            elif operation == 'adaptive':
                self.current_image = self.image_processor.apply_threshold(method='adaptive')
            elif operation == 'sobel':
                self.current_image = self.image_processor.detect_edges_sobel()
            
            self.display_images()
            self.log(f"Processing Applied: {operation}")
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
    def load_image_from_db(self):
        try:
            self.db.connect()
            # Exemple : récupérer le dernier enregistrement d'image
            result = self.db.cursor.execute("SELECT image_path FROM images ORDER BY id DESC LIMIT 1").fetchone()
            self.db.disconnect()

            if result:
                file_path = result[0]  # chemin stocké dans la DB
                self.image_processor.load_image(file_path)
                self.original_image = self.image_processor.original_image.copy()
                self.current_image = self.original_image.copy()
                self.display_images()
                self.log(f"Image Loaded from DB: {file_path}")
            else:
                messagebox.showinfo("Info", "No image found in database.")
        except Exception as e:
            self.log(f"Error loading image from DB: {str(e)}")
            messagebox.showerror("Error", f"Unable to load image from DB: {str(e)}")

    def reset_image(self):
        #Réinitialise l'image
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.image_processor.reset_to_original()
            self.display_images()
            self.log("Image Reset")
    
    def save_processed_image(self):
        #Sauvegarde l'image traitée
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
            )
            if file_path:
                cv2.imwrite(file_path, self.current_image)
                self.log(f"Image Saved: {file_path}")
    
    def store_image_metadata(self):
        #stock les métadonnées
        if self.image_processor.image_path:
            try:
                self.image_processor.store_metadata()
                self.log("Metadata Stored in Database")
                messagebox.showinfo("Success", "Metadata Saved!")
            except Exception as e:
                self.log(f"Error: {str(e)}")
                
    def store_processed_image_metadata(self):
        if self.current_image is not None:
            import time
        # Génère un nom unique
            file_path = f"processed_{int(time.time())}.png"

        # Sauvegarde l'image
            cv2.imwrite(file_path, self.current_image)

        # Mettre à jour le chemin dans l'image processor
            self.image_processor.image_path = file_path

        # Stocker le chemin et métadonnées dans la base
            self.image_processor.store_metadata()

        # Optionnel : aussi stocker dans la table images pour le chargement depuis DB
            try:
                self.db.connect()
                self.db.cursor.execute(
                    "INSERT INTO images (image_path, metadata) VALUES (?, ?)",
                    (file_path, "Processed Image")
                    )
                self.db.connection.commit()
                self.db.disconnect()
            except Exception as e:
                self.log(f"Error storing image in DB: {e}")

            self.log(f"Processed image saved: {file_path}")
            messagebox.showinfo("Success", "Processed image metadata saved!")

    def update_blur_kernel(self, value):
        kernel = int(round(float(value)))

    # forcer impair
        if kernel % 2 == 0:
            kernel += 1

    # bornes de sécurité
        kernel = max(3, min(kernel, 21))

    # afficher la vraie valeur utilisée
        self.blur_kernel_label.config(text=str(kernel))

    # stocker la valeur réelle pour le traitement
        self.current_blur_kernel = kernel



   
    
    #AUTRES
    
    def show_about(self):
        messagebox.showinfo(
            "About",
            "Environmental Data Processing\n\n"
            "Version 1.0\n"
            "Date: December 31, 2025\n\n"
            "Features::\n"
            "• SQLite database management\n"
            "• Data loading and filtering\n"
            "• Correlation analysis\n"
            "• Spectral Analysis (FFT)\n"
            "• Image Processing\n"
            "• Data Visualization\n"
        )


def main():
    root = tk.Tk()
    app = EnvironmentalDataGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
