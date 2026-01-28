import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from scipy.signal import welch     #calcul DSP
from scipy import fft
import cv2
from PIL import Image, ImageTk
import os
import time
from datetime import datetime

#Import des modules du projet
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
        self.root.minsize(1000, 600)  
        
        #variables
        self.db = AirQualityDatabase("db_air_quality")
        self.data = None
        self.COLUMN_MAP = {
            'CO': 'co_gt',
            'NO2': 'no2_gt',
            'Temperature': 'temperature',
            'Humidity': 'humidity'
}
        self.display_columns = list(self.COLUMN_MAP.keys())  


        self.current_image = None
        self.original_image = None
        self.image_processor = ImageProcessor()
        
        self.initialize_database()
    
        self.setup_style()
        
        self.create_menu()
        self.create_main_layout()
        
        self.data = pd.DataFrame()  
        self.update_stats()
        
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
        self.notebook.configure(takefocus=0) 

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
        
        columns = ('ID', 'Date', 'Time', 'CO', 'NO2', 'Temperature', 'Humidity')
        self.data_tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            if col in ['Date', 'Time']:
                self.data_tree.column(col, width=90)
            else:
                self.data_tree.column(col, width=80)
        
        #scrollbars
        vsb = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        hsb = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        #BOuttons for CRUD operations
        btn_frame = ttk.Frame(right_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        
        ttk.Button(btn_frame, text="Add Row", command=self.add_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit Row", command=self.edit_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Row", command=self.delete_row).pack(side=tk.LEFT, padx=5)
        
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

        content_frame = ttk.Frame(tab)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        content_frame.grid_columnconfigure(0, weight=1, uniform="equal")
        content_frame.grid_columnconfigure(1, weight=1, uniform="equal")
        content_frame.grid_rowconfigure(0, weight=1)

        filtered_frame = ttk.LabelFrame(content_frame, text="Filtered Data Comparison", padding=10)
        filtered_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        #Treeview
        filter_columns = ('ID', 'Original', 'Filtered')
        self.filtered_tree = ttk.Treeview(filtered_frame, columns=filter_columns, show='headings', height=15)
        self.filtered_tree.heading('ID', text='ID')
        self.filtered_tree.heading('Original', text='Original Value')
        self.filtered_tree.heading('Filtered', text='Filtered Value')
        self.filtered_tree.column('ID', width=80)
        self.filtered_tree.column('Original', width=120)
        self.filtered_tree.column('Filtered', width=120)

        #Scrollbars
        vsb_f = ttk.Scrollbar(filtered_frame, orient=tk.VERTICAL, command=self.filtered_tree.yview)
        hsb_f = ttk.Scrollbar(filtered_frame, orient=tk.HORIZONTAL, command=self.filtered_tree.xview)
        self.filtered_tree.configure(yscrollcommand=vsb_f.set, xscrollcommand=hsb_f.set)

        self.filtered_tree.grid(row=0, column=0, sticky='nsew')
        vsb_f.grid(row=0, column=1, sticky='ns')
        hsb_f.grid(row=1, column=0, sticky='ew')

        filtered_frame.grid_rowconfigure(0, weight=1)
        filtered_frame.grid_columnconfigure(0, weight=1)

        right_frame = ttk.LabelFrame(content_frame, text="Preview", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        self.filter_fig = Figure(figsize=(5, 5), dpi=90)
        self.filter_canvas = FigureCanvasTkAgg(self.filter_fig, right_frame)
        self.filter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
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
    
    #ONGLET 4: ANALYSE SPECTRALE
    
    def create_spectral_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Spectrum Analysis")
        
        # Conteneur pour la sidebar avec scroll
        sidebar_container = ttk.Frame(tab, width=220)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_container.pack_propagate(False)  # Garder la largeur fixe
        
        #canvas pour le scroll
        spectral_canvas_scroll = tk.Canvas(sidebar_container, highlightthickness=0, width=200)
        spectral_scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=spectral_canvas_scroll.yview)
        
        #Frame scrollable à l'intérieur du canvas
        left_frame = ttk.LabelFrame(spectral_canvas_scroll, text="Controls", padding=10)
        
        spectral_canvas_scroll.configure(yscrollcommand=spectral_scrollbar.set)
        
        #pack scrollbar et canvas
        spectral_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        spectral_canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        spectral_canvas_frame = spectral_canvas_scroll.create_window((0, 0), window=left_frame, anchor="nw")
        
        #Fonction pour mettre à jour la région de scroll
        def configure_spectral_scroll_region(event):
            spectral_canvas_scroll.configure(scrollregion=spectral_canvas_scroll.bbox("all"))
        
        def configure_spectral_canvas_width(event):
            spectral_canvas_scroll.itemconfig(spectral_canvas_frame, width=event.width)
        
        left_frame.bind("<Configure>", configure_spectral_scroll_region)
        spectral_canvas_scroll.bind("<Configure>", configure_spectral_canvas_width)
        
        #activer le scroll avec la molette de la souris
        def on_spectral_mousewheel(event):
            spectral_canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_spectral_mousewheel(event):
            spectral_canvas_scroll.bind_all("<MouseWheel>", on_spectral_mousewheel)
        
        def unbind_spectral_mousewheel(event):
            spectral_canvas_scroll.unbind_all("<MouseWheel>")
        
        spectral_canvas_scroll.bind("<Enter>", bind_spectral_mousewheel)
        spectral_canvas_scroll.bind("<Leave>", unbind_spectral_mousewheel)
        
        ttk.Label(left_frame, text="Variable", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.spectral_var = ttk.Combobox(left_frame, values=self.display_columns, state='readonly', width=20)
        self.spectral_var.set(self.display_columns[0])
        self.spectral_var.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(left_frame, text="Spectrum tools", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        self.spectral_repr_type = ttk.Combobox(left_frame, values=['FFT Fast Fourier Transform', 'Power Spectrum'], state='readonly', width=20)
        self.spectral_repr_type.set('Power Spectrum')
        self.spectral_repr_type.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(left_frame, text="Frequency Filters", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        self.spectral_filter_type = ttk.Combobox(left_frame, values=['No Filter', 'Low-pass', 'High-pass', 'Band-pass', 'Band-stop'], state='readonly', width=20)
        self.spectral_filter_type.set('No Filter')
        self.spectral_filter_type.pack(fill=tk.X, pady=5)
        self.spectral_filter_type.bind('<<ComboboxSelected>>', self._update_filter_fields)
        
        #Filter parameters frame
        self.filter_params_frame = ttk.Frame(left_frame)
        self.filter_params_frame.pack(fill=tk.X, pady=5)
        
        #cutoff frequency for low-pass and high-pass
        self.cutoff_frame = ttk.Frame(self.filter_params_frame)
        self.cutoff_frame.pack(fill=tk.X)
        ttk.Label(self.cutoff_frame, text="Cutoff (Hz):").pack(side=tk.LEFT)
        self.cutoff_freq = ttk.Entry(self.cutoff_frame, width=10)
        self.cutoff_freq.insert(0, "0.04")
        self.cutoff_freq.pack(side=tk.RIGHT, padx=5)
        
        # Band-pass frame (hidden by default)
        self.bandpass_frame = ttk.Frame(self.filter_params_frame)
        ttk.Label(self.bandpass_frame, text="Low (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.low_cutoff = ttk.Entry(self.bandpass_frame, width=8)
        self.low_cutoff.insert(0, "0.01")
        self.low_cutoff.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.bandpass_frame, text="High (Hz):").grid(row=1, column=0, sticky=tk.W)
        self.high_cutoff = ttk.Entry(self.bandpass_frame, width=8)
        self.high_cutoff.insert(0, "0.1")
        self.high_cutoff.grid(row=1, column=1, padx=5, pady=2)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Analysis", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        ttk.Button(left_frame, text="Run Analysis", command=self.run_spectral_analysis).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Reset", command=self.reset_spectral_analysis).pack(fill=tk.X, pady=5)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Results", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.spectral_results = scrolledtext.ScrolledText(left_frame, height=6, width=25, font=('Consolas', 9))
        self.spectral_results.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Save to Database", command=self.save_spectral_results).pack(fill=tk.X, pady=10)
        
        right_frame = ttk.LabelFrame(tab, text="Frequency Analysis", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.spectral_fig = Figure(figsize=(6, 4), dpi=100)
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_fig, right_frame)
        self.spectral_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
    
    def _update_filter_fields(self, event=None):
        filter_type = self.spectral_filter_type.get()
        
        # Hide all frames first
        self.cutoff_frame.pack_forget()
        self.bandpass_frame.pack_forget()
        
        # Show appropriate frame
        if filter_type == 'No Filter':
            #No parameters needed
            pass
        elif filter_type in ['Low-pass', 'High-pass']:
            self.cutoff_frame.pack(fill=tk.X)
        elif filter_type in ['Band-pass', 'Band-stop']:
            self.bandpass_frame.pack(fill=tk.X)
    
    #ONGLET 5: TRAITEMENT D'IMAGES
    
    def create_image_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Image Processing")

        sidebar_container = ttk.Frame(tab, width=220)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_container.pack_propagate(False)  # Garder la largeur fixe
        
        canvas = tk.Canvas(sidebar_container, highlightthickness=0, width=200)
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=canvas.yview)
        
        left_frame = ttk.LabelFrame(canvas, text="Processing", padding=10)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_frame = canvas.create_window((0, 0), window=left_frame, anchor="nw")
        
        # Fonction pour mettre à jour la région de scroll
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        
        left_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)

        ttk.Button(left_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Load Image from DB", command=self.load_image_from_db).pack(fill=tk.X, pady=3)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        ttk.Label(left_frame, text="Processing:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        self.processing_options = {
            'Grayscale Conversion': 'grayscale',
            'Gaussian Blur': 'blur',
            'Canny Edge Detection': 'canny',
            'Otsu Thresholding': 'otsu',
            'Adaptive Thresholding': 'adaptive',
            'Sobel Filter': 'sobel'
        }
        
        self.processing_combo = ttk.Combobox(
            left_frame, 
            values=list(self.processing_options.keys()),
            state='readonly',
            width=22
        )
        self.processing_combo.set('Grayscale Conversion')
        self.processing_combo.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Apply Processing", command=self.apply_selected_processing).pack(fill=tk.X, pady=3)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        ttk.Label(left_frame, text="Parameters:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        ttk.Label(left_frame, text="Blur Kernel:").pack(anchor=tk.W, pady=(5, 0))
        
        kernel_frame = ttk.Frame(left_frame)
        kernel_frame.pack(fill=tk.X)
        
        self.blur_kernel_label = ttk.Label(kernel_frame, text="5", width=3)
        self.current_blur_kernel = 5
        
        self.blur_kernel = ttk.Scale(
            kernel_frame,
            from_=3,
            to=21,
            orient=tk.HORIZONTAL,
            command=self.update_blur_kernel
        )
        self.blur_kernel.set(5)
        self.blur_kernel.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.blur_kernel_label.pack(side=tk.RIGHT, padx=5)

        ttk.Label(left_frame, text="Canny Threshold 1:").pack(anchor=tk.W, pady=(5, 0))
        
        thresh1_frame = ttk.Frame(left_frame)
        thresh1_frame.pack(fill=tk.X)
        
        self.canny_thresh1 = ttk.Scale(thresh1_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.canny_thresh1.set(50)
        self.canny_thresh1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.canny_thresh1_label = ttk.Label(thresh1_frame, text="50", width=3)
        self.canny_thresh1_label.pack(side=tk.RIGHT, padx=5)
        self.canny_thresh1.configure(command=lambda v: self.canny_thresh1_label.configure(text=f"{int(float(v))}"))
        
        ttk.Label(left_frame, text="Canny Threshold 2:").pack(anchor=tk.W, pady=(5, 0))
        
        thresh2_frame = ttk.Frame(left_frame)
        thresh2_frame.pack(fill=tk.X)
        
        self.canny_thresh2 = ttk.Scale(thresh2_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.canny_thresh2.set(150)
        self.canny_thresh2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.canny_thresh2_label = ttk.Label(thresh2_frame, text="150", width=3)
        self.canny_thresh2_label.pack(side=tk.RIGHT, padx=5)
        self.canny_thresh2.configure(command=lambda v: self.canny_thresh2_label.configure(text=f"{int(float(v))}"))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        ttk.Button(left_frame, text="Reset", command=self.reset_image).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Save", command=self.save_processed_image).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Store Metadata", command=self.store_processed_image_metadata).pack(fill=tk.X, pady=3)
        
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
    
    def apply_selected_processing(self):
        selected = self.processing_combo.get()
        if selected in self.processing_options:
            processing_key = self.processing_options[selected]
            self.apply_image_processing(processing_key)
      
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
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.data is not None and len(self.data) > 0:
            for idx, row in self.data.iterrows():
                values = (
                    row.get('id', idx),
                    row.get('date', '') if pd.notna(row.get('date')) else '',
                    row.get('time', '') if pd.notna(row.get('time')) else '',
                    f"{row.get('co_gt', 0):.2f}" if pd.notna(row.get('co_gt')) else '',
                    f"{row.get('no2_gt', 0):.2f}" if pd.notna(row.get('no2_gt')) else '',
                    f"{row.get('temperature', 0):.1f}" if pd.notna(row.get('temperature')) else '',
                    f"{row.get('humidity', 0):.1f}" if pd.notna(row.get('humidity')) else ''
                )
                self.data_tree.insert('', tk.END, values=values)
    
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
                self.db.connect()
                self.db.create_tables()
                self.db.disconnect()
                
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
                # Reset auto-increment for MySQL
                self.db.cursor.execute("ALTER TABLE air_quality_measurements AUTO_INCREMENT = 1")
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
    
    def add_row(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Row")
        dialog.geometry("300x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        fields = ['Date', 'Time', 'CO', 'NO2', 'Temperature', 'Humidity']
        entries = {}
        
        for i, field in enumerate(fields):
            ttk.Label(dialog, text=f"{field}:").grid(row=i, column=0, padx=10, pady=5, sticky='w')
            entry = ttk.Entry(dialog)
            if field == 'Date':
                entry.insert(0, datetime.now().strftime('%d/%m/%Y'))
            elif field == 'Time':
                entry.insert(0, datetime.now().strftime('%H.%M.%S'))
            entry.grid(row=i, column=1, padx=10, pady=5, sticky='ew')
            entries[field] = entry
        
        def save_new_row():
            try:
                self.db.connect()
                date_val = entries['Date'].get() if entries['Date'].get() else 'N/A'
                time_val = entries['Time'].get() if entries['Time'].get() else 'N/A'
                co = float(entries['CO'].get()) if entries['CO'].get() else None
                no2 = float(entries['NO2'].get()) if entries['NO2'].get() else None
                temp = float(entries['Temperature'].get()) if entries['Temperature'].get() else None
                hum = float(entries['Humidity'].get()) if entries['Humidity'].get() else None
                
                self.db.insert_measurement(
                    date=date_val, time=time_val,
                    co_gt=co, no2_gt=no2,
                    temperature=temp, humidity=hum
                )
                self.db.disconnect()
                self.load_data_from_db()
                self.log("New row added")
                dialog.destroy()
                messagebox.showinfo("Success", "Row added successfully!")
            except Exception as e:
                self.log(f"Error adding row: {e}")
                messagebox.showerror("Error", f"Failed to add row: {e}")
        
        ttk.Button(dialog, text="Save", command=save_new_row).grid(row=len(fields), column=0, columnspan=2, pady=20)
    
    def edit_row(self):
        selected = self.data_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a row to edit")
            return
        
        item = self.data_tree.item(selected[0])
        values = item['values']
        record_id = values[0]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Row")
        dialog.geometry("300x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        fields = ['Date', 'Time', 'CO', 'NO2', 'Temperature', 'Humidity']
        entries = {}
        
        field_indices = {'Date': 1, 'Time': 2, 'CO': 3, 'NO2': 4, 'Temperature': 5, 'Humidity': 6}
        
        for i, field in enumerate(fields):
            ttk.Label(dialog, text=f"{field}:").grid(row=i, column=0, padx=10, pady=5, sticky='w')
            entry = ttk.Entry(dialog)
            value_idx = field_indices[field]
            entry.insert(0, str(values[value_idx]) if values[value_idx] else '')
            entry.grid(row=i, column=1, padx=10, pady=5, sticky='ew')
            entries[field] = entry
        
        def save_edit():
            try:
                self.db.connect()
                updates = {}
                if entries['Date'].get():
                    updates['date'] = entries['Date'].get()
                if entries['Time'].get():
                    updates['time'] = entries['Time'].get()
                if entries['CO'].get():
                    updates['co_gt'] = float(entries['CO'].get())
                if entries['NO2'].get():
                    updates['no2_gt'] = float(entries['NO2'].get())
                if entries['Temperature'].get():
                    updates['temperature'] = float(entries['Temperature'].get())
                if entries['Humidity'].get():
                    updates['humidity'] = float(entries['Humidity'].get())
                
                if updates:
                    self.db.update_measurement(record_id, **updates)
                self.db.disconnect()
                self.load_data_from_db()
                self.log(f"Row {record_id} updated")
                dialog.destroy()
                messagebox.showinfo("Success", "Row updated successfully!")
            except Exception as e:
                self.log(f"Error editing row: {e}")
                messagebox.showerror("Error", f"Failed to edit row: {e}")
        
        ttk.Button(dialog, text="Save", command=save_edit).grid(row=len(fields), column=0, columnspan=2, pady=20)
    
    def delete_row(self):
        selected = self.data_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a row to delete")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this row?"):
            try:
                item = self.data_tree.item(selected[0])
                record_id = item['values'][0]
                
                self.db.connect()
                self.db.delete_measurement(record_id)
                self.db.disconnect()
                
                self.load_data_from_db()
                self.log(f"Row {record_id} deleted")
                messagebox.showinfo("Success", "Row deleted successfully!")
            except Exception as e:
                self.log(f"Error deleting row: {e}")
                messagebox.showerror("Error", f"Failed to delete row: {e}")
    
    def export_data(self):
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
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        var_selected = self.filter_var.get()
        filter_type = self.filter_type.get()
        df_column = self.COLUMN_MAP[var_selected]
        
        # Extraire les données originales (sans NaN)
        original_series = self.data[df_column].dropna()
        original = original_series.values
        
        if filter_type == 'Moving Average':
            #filtrage par moyenne mobile centrée
            window = int(float(self.window_size.get()))
            #appliquer la moyenne mobile centrée
            filtered_series = pd.Series(original).rolling(window=window, center=True).mean()
            #remplir les valeurs NaN aux extrémités
            filtered_series = filtered_series.fillna(method='bfill').fillna(method='ffill')
            filtered = filtered_series.values
            title = f"Moving Average (window={window})"
            filter_desc = f"Moving Average with window size {window}"
            
        elif filter_type == 'Threshold Filter':
            try:
                min_val = float(self.threshold_min.get())
                max_val = float(self.threshold_max.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values for the thresholds.")
                return
            
            if min_val >= max_val:
                messagebox.showerror("Error", "The minimum threshold must be less than the maximum threshold.")
                return
            
            filtered = np.clip(original, min_val, max_val)
            title = f"Thresholding [{min_val}, {max_val}]"
            filter_desc = f"Thresholding between {min_val} et {max_val}"
        else:
            messagebox.showwarning("Warning", "Unrecognized filter type")
            return
        
        #stocker les données filtrées 
        self.filtered_data = filtered
        self.original_data = original
        
        #AFFICHAGE DU TABLEAU COMPARATIF 
        for item in self.filtered_tree.get_children():
            self.filtered_tree.delete(item)
        
        for i in range(len(original)):
            values = (
                i,
                f"{original[i]:.4f}",
                f"{filtered[i]:.4f}"
            )
            self.filtered_tree.insert('', tk.END, values=values)
        
        self.filter_fig.clear()
        ax = self.filter_fig.add_subplot(111)
        
        #tracer les séries (limiter à 500)
        plot_limit = min(500, len(original))
        x_axis = range(plot_limit)
        
        ax.plot(x_axis, original[:plot_limit], 'b-', linewidth=1.5, alpha=0.6, label='Original')
        
        ax.plot(x_axis, filtered[:plot_limit], 'r-', linewidth=2, alpha=0.8, label='Filtered')
        
        # Mise en forme du graphique
        ax.set_xlabel('ID', fontsize=9)
        ax.set_ylabel(var_selected, fontsize=9)
        ax.set_title(f'{var_selected}: {title}', fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        self.filter_fig.tight_layout()
        self.filter_canvas.draw()
        
        self.log(f"Filter Applied: {filter_type} to {var_selected} - {filter_desc}")
    
    def reset_filter(self):
        self.filter_fig.clear()
        self.filter_canvas.draw()
        
        for item in self.filtered_tree.get_children():
            self.filtered_tree.delete(item)
        
        self.filtered_data = None
        self.original_data = None
        
        self.log("Filtering Reset")
        
    def save_filtered_data(self):
        if self.data is None or len(self.data) == 0:
            messagebox.showwarning("Warning", "No data loaded to save")
            return
        
        #vérifier si des données filtrées existent
        if not hasattr(self, 'filtered_data') or self.filtered_data is None:
            messagebox.showwarning("Warning", "Please apply a filter first")
            return
        
        var_selected = self.filter_var.get()
        df_column = self.COLUMN_MAP[var_selected]
        filter_type = self.filter_type.get()
        
        #récupérer les paramètres du filtre
        window_size = int(self.window_size.get())
        try:
            threshold_min = float(self.threshold_min.get())
        except:
            threshold_min = None
        try:
            threshold_max = float(self.threshold_max.get())
        except:
            threshold_max = None
        
        try:
            self.db.connect()
            
            #Get the indices of non-null values in the original column
            original_series = self.data[df_column].dropna()
            
            #Stocker chaque ligne filtrée dans l'historq
            saved_count = 0
            for i, (idx, original_value) in enumerate(original_series.items()):
                if i < len(self.filtered_data):
                    # Get the record id for this row
                    record_id = self.data.loc[idx, 'id']
                    filtered_value = float(self.filtered_data[i])
                    
                    #insérer dans l'historique des données filtrées
                    self.db.insert_filtered_data(
                        original_record_id=int(record_id),
                        variable_name=var_selected,
                        filter_type=filter_type,
                        window_size=window_size,
                        threshold_min=threshold_min,
                        threshold_max=threshold_max,
                        original_value=float(original_value),
                        filtered_value=filtered_value,
                        row_index=int(idx)
                    )
                    saved_count += 1
            
            self.db.disconnect()
            
            filter_params = f"Filter: {filter_type}"
            if filter_type == 'Moving Average':
                filter_params += f", Window: {window_size}"
            else:
                filter_params += f", Min: {threshold_min}, Max: {threshold_max}"
            
            self.log(f"Filtered data saved: {saved_count} records for {var_selected} ({filter_params})")
            messagebox.showinfo(
                "Success", 
                f"Filtered data saved to history!\n\n"
                f"Variable: {var_selected}\n"
                f"Filter Type: {filter_type}\n"
                f"Records saved: {saved_count}\n\n"
                f"All filtered data has been stored in the database history."
            )
            
        except Exception as e:
            self.log(f"Error saving filtered data: {e}")
            messagebox.showerror("Error", f"Unable to save filtered data: {e}")
            if self.db.connection:
                self.db.disconnect()
    
 
  
    #FONCTIONS CORRÉLATIONS
    
    def show_correlation_heatmap(self):
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
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
        
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontweight='bold')
        
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()
        
        self.log(f"Correlation Heatmap Generated ({method})")
    
    def show_scatter_plot(self):
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
        
        display_x = self.corr_var_x.get()
        display_y = self.corr_var_y.get()
        
        ax.set_xlabel(display_x)
        ax.set_ylabel(display_y)
        ax.set_title(f'{display_x} vs {display_y} (r = {corr:.3f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()
        
        self.corr_result_label.configure(text=f"Correlation:\nr = {corr:.4f}")
        self.log(f"Scatter plot: {var_x} vs {var_y} (r={corr:.3f})")
    
    def save_correlations(self):
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
        from scipy.signal import butter, sosfilt
        
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        try:
            var_selected = self.spectral_var.get()
            repr_type = self.spectral_repr_type.get()
            var_fft = self.COLUMN_MAP[var_selected]
            signal_original = self.data[var_fft].dropna().values.copy()
            signal_filtered = signal_original.copy()
            
            if len(signal_original) < 10:
                messagebox.showwarning("Warning", "Not enough data points for analysis")
                return
            
            #remove mean (DC component)
            signal_original = signal_original - np.mean(signal_original)
            signal_filtered = signal_filtered - np.mean(signal_filtered)
            
            #APPLY FILTERS 
            filter_info = ""
            filter_type = self.spectral_filter_type.get()
            show_comparison = False
            
            fs = 1.0  #sampling freq (1 sample per hour)
            nyq = 0.5 * fs  #Nyquist freq
            order = 3
            
            try:
                if filter_type == 'No Filter':
                    filter_info = ""
                    signal_filtered = signal_original.copy()
                    
                elif filter_type == 'Low-pass':
                    cutoff = float(self.cutoff_freq.get())
                    if cutoff <= 0 or cutoff >= nyq:
                        messagebox.showwarning("Warning", f"Cutoff must be between 0 and {nyq} Hz")
                        return
                    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
                    signal_filtered = sosfilt(sos, signal_filtered)
                    filter_info = f" + Low-pass ({cutoff} Hz)"
                    show_comparison = True
                    
                elif filter_type == 'High-pass':
                    cutoff = float(self.cutoff_freq.get())
                    if cutoff <= 0 or cutoff >= nyq:
                        messagebox.showwarning("Warning", f"Cutoff must be between 0 and {nyq} Hz")
                        return
                    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
                    signal_filtered = sosfilt(sos, signal_filtered)
                    filter_info = f" + High-pass ({cutoff} Hz)"
                    show_comparison = True
                    
                elif filter_type == 'Band-pass':
                    low_cut = float(self.low_cutoff.get())
                    high_cut = float(self.high_cutoff.get())
                    if low_cut <= 0 or high_cut >= nyq or low_cut >= high_cut:
                        messagebox.showwarning("Warning", f"Frequencies must satisfy: 0 < low < high < {nyq}")
                        return
                    sos = butter(order, [low_cut, high_cut], btype='band', fs=fs, output='sos')
                    signal_filtered = sosfilt(sos, signal_filtered)
                    filter_info = f" + Band-pass ({low_cut}-{high_cut} Hz)"
                    show_comparison = True
                elif filter_type == 'Band-stop':
                    low_cut = float(self.low_cutoff.get())
                    high_cut = float(self.high_cutoff.get())
                    if low_cut <= 0 or high_cut >= nyq or low_cut >= high_cut:
                        messagebox.showwarning("Warning", f"Frequencies must satisfy: 0 < low < high < {nyq}")
                        return
                    sos = butter(order, [low_cut, high_cut], btype='bandstop', fs=fs, output='sos')
                    signal_filtered = sosfilt(sos, signal_filtered)
                    filter_info = f" + Band-stop ({low_cut}-{high_cut} Hz)"
                    show_comparison = True
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid filter parameters: {e}")
                return
            
            signal = signal_filtered
            
            #SPECTRAL ANALYSIS ON FILTERED SIGNAL
            self.spectral_fig.clear()
            
            ax1 = self.spectral_fig.add_subplot(211)
            ax2 = self.spectral_fig.add_subplot(212)
            
            display_len = min(500, len(signal))
            ax1.plot(signal_original[:display_len], 'b-', linewidth=0.7, alpha=0.5, label='Original' if show_comparison else '')
            ax1.plot(signal_filtered[:display_len], 'r-', linewidth=0.8, label='Filtered' if show_comparison else '')
            ax1.set_title(f'Time Signal: {var_selected}{filter_info}', fontweight='bold')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Value')
            if show_comparison:
                ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            if repr_type == 'FFT Fast Fourier Transform':
                n = len(signal)
                fft_result = fft.fft(signal)
                frequencies = fft.fftfreq(n, d=1.0)
                positive_mask = frequencies >= 0
                frequencies = frequencies[positive_mask]
                amplitudes = np.abs(fft_result[positive_mask]) * 2 / n
                
                plot_limit = len(frequencies) // 2
                ax2.plot(frequencies[:plot_limit], amplitudes[:plot_limit], 'r-', linewidth=1)
                ax2.axvline(x=1/24, color='green', linestyle='--', alpha=0.7, label='Daily (24h)')
                ax2.axvline(x=1/168, color='orange', linestyle='--', alpha=0.7, label='Weekly (168h)')
                ax2.set_title(f'Fast Fourier Transform: {var_selected}{filter_info}', fontweight='bold')
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Amplitude')
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
                
                #dominant frequencies
                peak_idx = np.argsort(amplitudes[:plot_limit])[-5:][::-1]
                results = f"Fast Fourier Transform Analysis: {var_selected}{filter_info}\n"
                results += "-" * 40 + "\n"
                results += "Dominant Frequencies:\n\n"
                for idx in peak_idx:
                    f = frequencies[idx]
                    if f > 0:
                        period = 1/f
                        results += f"• {f:.5f} Hz - Amplitude: {amplitudes[idx]:.4f}\n"
                        results += f"  Period: {period:.1f}h\n\n"
                        
            else:  
               #POWER SPECTRUM 
                nperseg = min(256, len(signal)//4)
                if nperseg < 4:
                    nperseg = len(signal)
                    
                frequencies, power = welch(signal, fs=1.0, nperseg=nperseg)
                
                ax2.semilogy(frequencies, power, 'r-', linewidth=1)
                ax2.axvline(x=1/24, color='green', linestyle='--', alpha=0.7, label='Daily (24h)')
                ax2.axvline(x=1/168, color='orange', linestyle='--', alpha=0.7, label='Weekly (168h)')
                ax2.set_title(f'Power Spectrum: {var_selected}{filter_info}', fontweight='bold')
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Power Spectral Density')
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3, which='both')
                
                #dominant frequencies
                peak_idx = np.argsort(power)[-5:][::-1]
                results = f"Power Spectrum Analysis: {var_selected}{filter_info}\n"
                results += "-" * 40 + "\n"
                results += "Dominant Frequencies:\n\n"
                for idx in peak_idx:
                    f = frequencies[idx]
                    if f > 0:
                        period = 1/f
                        results += f"• {f:.5f} Hz - Power: {power[idx]:.4e}\n"
                        results += f"  Period: {period:.1f}h\n\n"
            
            self.spectral_results.delete(1.0, tk.END)
            self.spectral_results.insert(tk.END, results)
            
            self.spectral_fig.tight_layout()
            self.spectral_canvas.draw()
            self.log(f"Spectral Analysis ({repr_type}): {var_selected}{filter_info}")
            
        except Exception as e:
            self.log(f"Error in spectral analysis: {e}")
            messagebox.showerror("Error", f"Spectral analysis failed: {e}")
    
    def save_spectral_results(self):
        try:
            var_selected = self.spectral_var.get()
            var_column = self.COLUMN_MAP[var_selected]
            analyzer = SpectralAnalyzer()
            analyzer.load_data()
            analyzer.store_spectral_results(var_column)
            self.log("Spectral Results Saved")
            messagebox.showinfo("Success", "Results Saved!")
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def reset_spectral_analysis(self):
        self.spectral_fig.clear()
        self.spectral_canvas.draw()
        
        #clear results text
        self.spectral_results.delete(1.0, tk.END)
        
        # Reset filter type to default
        self.spectral_filter_type.set('No Filter')
        self._update_filter_fields()
        
        # Reset representation type
        self.spectral_repr_type.set('Power Spectrum')
        
        # Reset variable to first option
        self.spectral_var.set(self.display_columns[0])
        
        # Reset filter parameters
        self.cutoff_freq.delete(0, tk.END)
        self.cutoff_freq.insert(0, "0.04")
        self.low_cutoff.delete(0, tk.END)
        self.low_cutoff.insert(0, "0.01")
        self.high_cutoff.delete(0, tk.END)
        self.high_cutoff.insert(0, "0.1")
        
        self.log("Spectral Analysis Reset")
    
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
        if self.original_image is not None:
            img_orig = self.resize_image_for_display(self.original_image)
            self.orig_photo = ImageTk.PhotoImage(img_orig)
            self.orig_image_label.configure(image=self.orig_photo)
        
        if self.current_image is not None:
            img_proc = self.resize_image_for_display(self.current_image)
            self.proc_photo = ImageTk.PhotoImage(img_proc)
            self.proc_image_label.configure(image=self.proc_photo)
    
    def resize_image_for_display(self, img, max_size=400):
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
            self.db.cursor.execute(
                "SELECT id, filename, file_path, width, height, processing_methods, created_at FROM image_metadata ORDER BY created_at DESC"
            )
            results = self.db.cursor.fetchall()
            self.db.disconnect()

            if not results:
                messagebox.showinfo("Info", "No images found in database.\nPlease store an image first using 'Store Metadata'.")
                return

            #créer une fenetre de sélection
            selection_dialog = tk.Toplevel(self.root)
            selection_dialog.title("Select Image from Database")
            selection_dialog.geometry("700x400")
            selection_dialog.transient(self.root)
            selection_dialog.grab_set()
            
            #centrer la fenetre
            selection_dialog.update_idletasks()
            x = (selection_dialog.winfo_screenwidth() - 700) // 2
            y = (selection_dialog.winfo_screenheight() - 400) // 2
            selection_dialog.geometry(f"700x400+{x}+{y}")

            main_frame = ttk.Frame(selection_dialog, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(main_frame, text="Select an image to load:", style="Header.TLabel").pack(pady=(0, 10))

            columns = ('ID', 'Filename', 'Size', 'Processing', 'Date')
            tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=12)
            
            tree.heading('ID', text='ID')
            tree.heading('Filename', text='Filename')
            tree.heading('Size', text='Dimensions')
            tree.heading('Processing', text='Processing Applied')
            tree.heading('Date', text='Date Added')
            
            tree.column('ID', width=50, anchor='center')
            tree.column('Filename', width=200)
            tree.column('Size', width=100, anchor='center')
            tree.column('Processing', width=200)
            tree.column('Date', width=120, anchor='center')

            scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            #remplir le treeview
            image_data = {}  #stocker les données pour la sélection
            for row in results:
                img_id, filename, file_path, width, height, processing, created_at = row
                size_str = f"{width}x{height}" if width and height else "N/A"
                processing_str = processing if processing else "Original"
                date_str = str(created_at)[:19] if created_at else "N/A"
                
                tree.insert('', tk.END, iid=str(img_id), values=(img_id, filename, size_str, processing_str, date_str))
                image_data[str(img_id)] = {'path': file_path, 'filename': filename}

            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            button_frame = ttk.Frame(selection_dialog, padding="10")
            button_frame.pack(fill=tk.X)

            def load_selected():
                selected = tree.selection()
                if not selected:
                    messagebox.showwarning("Warning", "Please select an image first.")
                    return
                
                img_id = selected[0]
                file_path = image_data[img_id]['path']
                filename = image_data[img_id]['filename']
                
                if not os.path.exists(file_path):
                    messagebox.showerror("Error", f"Image file not found:\n{file_path}\n\nThe file may have been moved or deleted.")
                    return
                
                try:
                    self.image_processor.load_image(file_path)
                    self.original_image = self.image_processor.original_image.copy()
                    self.current_image = self.original_image.copy()
                    self.display_images()
                    self.log(f"Image Loaded from DB: {filename}")
                    selection_dialog.destroy()
                    messagebox.showinfo("Success", f"Image '{filename}' loaded successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load image: {e}")

            def on_double_click(event):
                load_selected()

            tree.bind('<Double-1>', on_double_click)

            ttk.Button(button_frame, text="Load Selected", command=load_selected).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=selection_dialog.destroy, style="Secondary.TButton").pack(side=tk.RIGHT, padx=5)

        except Exception as e:
            self.log(f"Error loading image from DB: {str(e)}")
            messagebox.showerror("Error", f"Unable to load images from DB: {str(e)}")

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.image_processor.reset_to_original()
            self.display_images()
            self.log("Image Reset")
    
    def save_processed_image(self):
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
            )
            if file_path:
                cv2.imwrite(file_path, self.current_image)
                self.log(f"Image Saved: {file_path}")
    
    def store_image_metadata(self):
        if not self.image_processor.image_path:
            messagebox.showwarning("Warning", "No image loaded.\nPlease load an image first.")
            return
            
        try:
            #créer le dossier de stockage s'il n'existe pas
            storage_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "stored")
            os.makedirs(storage_folder, exist_ok=True)
            
            original_path = self.image_processor.image_path
            filename = os.path.basename(original_path)
            
            #si l'image n'est pas déjà dans le dossier de stockage, la copier
            if storage_folder not in os.path.abspath(original_path):
                #pour éviter les conflits de noms
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{int(time.time())}{ext}"
                new_path = os.path.join(storage_folder, new_filename)
                
                #copier l'image originale
                import shutil
                shutil.copy2(original_path, new_path)
                
                # Màj le chemin dans l'image processor
                self.image_processor.image_path = new_path
                self.log(f"Image copied to: {new_path}")
            
            self.image_processor.store_metadata()
            self.log("Metadata Stored in Database")
            messagebox.showinfo("Success", f"Image metadata saved!\n\nStored in: images/stored/")
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to store metadata: {str(e)}")
                
    def store_processed_image_metadata(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No processed image to save.\nPlease load and process an image first.")
            return
        
        storage_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "stored")
        os.makedirs(storage_folder, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"processed_{timestamp}.png"
        file_path = os.path.join(storage_folder, filename)

        try:
            #sauvegarde l'image dans le dossier dédié
            cv2.imwrite(file_path, self.current_image)

            #màj le chemin dans l'image processor
            self.image_processor.image_path = file_path
            self.image_processor.image = self.current_image.copy()
            
            #récupérer l'historique des traitements
            processing_methods = ", ".join(self.image_processor.processing_history) if self.image_processor.processing_history else "manual_processing"

            #stocker les métadonnées dans image_metadata
            self.image_processor.store_metadata()

            self.log(f"Processed image saved: {file_path}")
            messagebox.showinfo("Success", f"Processed image saved successfully!\n\nLocation: {file_path}")
            
        except Exception as e:
            self.log(f"Error storing processed image: {e}")
            messagebox.showerror("Error", f"Failed to save processed image: {e}")

    def update_blur_kernel(self, value):
        kernel = int(round(float(value)))

        if kernel % 2 == 0:
            kernel += 1

        kernel = max(3, min(kernel, 21))

        self.blur_kernel_label.config(text=str(kernel))

        self.current_blur_kernel = kernel
    
    #AUTRES
    
    def show_about(self):
        messagebox.showinfo(
            "About",
            "Environmental Data Processing\n\n"
            "Version 1.0\n"
            "Date: December 31, 2025\n\n"
            "Features::\n"
            "• MySQL database management\n"
            "• Data loading and filtering\n"
            "• Correlation analysis\n"
            "• Spectral Analysis (FFT)\n"
            "• Spectral Filters (Low/High/Band-pass)\n"
            "• Image Processing\n"
            "• Data Visualization\n"
        )


def main():
    root = tk.Tk()
    app = EnvironmentalDataGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
