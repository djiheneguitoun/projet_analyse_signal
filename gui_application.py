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
        self.db = AirQualityDatabase("db_air_quality")
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
        """Configure un style moderne et professionnel pour l'application."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # =================================================================
        # PALETTE DE COULEURS MODERNE (tons verts naturels)
        # =================================================================
        colors = {
            'bg_primary': '#F8FAF9',        # Fond principal (légèrement plus chaud)
            'bg_secondary': '#EDF3F0',      # Fond secondaire
            'bg_accent': '#E4EDE8',         # Fond accent
            'accent_primary': '#5B8A72',    # Vert principal (plus saturé)
            'accent_light': '#7DA894',      # Vert clair
            'accent_dark': '#3D6B54',       # Vert foncé
            'accent_hover': '#8FBEA8',      # Vert hover
            'text_primary': '#1E3329',      # Texte principal (plus foncé)
            'text_secondary': '#3D5347',    # Texte secondaire
            'text_muted': '#6B8579',        # Texte atténué
            'border': '#C8D9CF',            # Bordures
            'border_light': '#DDE8E2',      # Bordures légères
            'white': '#FFFFFF',
            'success': '#4A9B6E',           # Succès
            'warning': '#C9A227',           # Avertissement
        }
        
        # =================================================================
        # CONFIGURATION GLOBALE
        # =================================================================
        
        # Root window background
        self.root.configure(bg=colors['bg_primary'])
        
        # =================================================================
        # NOTEBOOK (Onglets) - Style moderne avec tabs arrondies
        # =================================================================
        style.configure(
            "TNotebook",
            background=colors['bg_primary'],
            borderwidth=0,
            tabmargins=[8, 8, 8, 0]
        )
        
        style.configure(
            "TNotebook.Tab",
            background=colors['bg_accent'],
            foreground=colors['text_secondary'],
            padding=[20, 10],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            focuscolor=""
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", colors['accent_primary']),
                ("active", colors['accent_light'])
            ],
            foreground=[
                ("selected", colors['white']),
                ("active", colors['text_primary'])
            ],
            padding=[
                ("selected", [20, 10]),
                ("active", [20, 10]),
                ("!selected", [20, 10])
            ],
            focuscolor=[("focus", "")]
        )
        
        # =================================================================
        # FRAMES
        # =================================================================
        style.configure(
            "TFrame",
            background=colors['bg_primary']
        )
        
        style.configure(
            "Card.TFrame",
            background=colors['white'],
            relief="flat"
        )

        style.configure(
            "TLabelframe",
            background=colors['bg_primary'],
            foreground=colors['accent_primary'],
            bordercolor=colors['border'],
            lightcolor=colors['border_light'],
            darkcolor=colors['border'],
            borderwidth=2,
            relief="groove"
        )

        style.configure(
            "TLabelframe.Label",
            background=colors['bg_primary'],
            foreground=colors['accent_dark'],
            font=("Segoe UI", 10, "bold"),
            padding=[5, 2]
        )
        
        # =================================================================
        # LABELS
        # =================================================================
        style.configure(
            "TLabel",
            background=colors['bg_primary'],
            foreground=colors['text_primary'],
            font=("Segoe UI", 10)
        )

        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 13, "bold"),
            foreground=colors['accent_dark'],
            background=colors['bg_primary']
        )
        
        style.configure(
            "SubHeader.TLabel",
            font=("Segoe UI", 11, "bold"),
            foreground=colors['text_secondary'],
            background=colors['bg_primary']
        )
        
        style.configure(
            "Muted.TLabel",
            font=("Segoe UI", 9),
            foreground=colors['text_muted'],
            background=colors['bg_primary']
        )
        
        # =================================================================
        # BOUTONS - Style moderne avec effets hover
        # =================================================================
        style.configure(
            "TButton",
            background=colors['accent_primary'],
            foreground=colors['white'],
            font=("Segoe UI", 10, "bold"),
            padding=[12, 8],
            borderwidth=0,
            focuscolor="",
            anchor="center"
        )

        style.map(
            "TButton",
            background=[
                ("pressed", colors['accent_dark']),
                ("active", colors['accent_hover']),
                ("disabled", colors['bg_accent'])
            ],
            foreground=[
                ("disabled", colors['text_muted'])
            ],
            relief=[
                ("pressed", "flat"),
                ("!pressed", "flat")
            ]
        )
        
        # Bouton secondaire (outline style)
        style.configure(
            "Secondary.TButton",
            background=colors['bg_primary'],
            foreground=colors['accent_primary'],
            font=("Segoe UI", 10),
            padding=[12, 8],
            borderwidth=2,
            relief="solid"
        )
        
        style.map(
            "Secondary.TButton",
            background=[
                ("active", colors['bg_accent']),
                ("pressed", colors['accent_light'])
            ],
            foreground=[
                ("pressed", colors['white'])
            ]
        )
        
        # Bouton accent/action
        style.configure(
            "Accent.TButton",
            background=colors['success'],
            foreground=colors['white'],
            font=("Segoe UI", 10, "bold"),
            padding=[12, 8]
        )
        
        style.map(
            "Accent.TButton",
            background=[
                ("active", "#5AAF7E"),
                ("pressed", "#3A8B5E")
            ]
        )
        
        # =================================================================
        # COMBOBOX - Style moderne
        # =================================================================
        style.configure(
            "TCombobox",
            background=colors['white'],
            foreground=colors['text_primary'],
            fieldbackground=colors['white'],
            selectbackground=colors['accent_light'],
            selectforeground=colors['white'],
            bordercolor=colors['border'],
            arrowcolor=colors['accent_primary'],
            padding=[8, 6],
            font=("Segoe UI", 10)
        )
        
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", colors['white']),
                ("disabled", colors['bg_secondary'])
            ],
            foreground=[
                ("readonly", colors['text_primary']),
                ("disabled", colors['text_muted'])
            ],
            background=[
                ("active", colors['accent_light']),
                ("pressed", colors['accent_primary'])
            ],
            bordercolor=[
                ("focus", colors['accent_primary']),
                ("hover", colors['accent_light'])
            ],
            arrowcolor=[
                ("disabled", colors['text_muted'])
            ]
        )
        
        # Style for dropdown list
        self.root.option_add('*TCombobox*Listbox.background', colors['white'])
        self.root.option_add('*TCombobox*Listbox.foreground', colors['text_primary'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', colors['accent_primary'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', colors['white'])
        self.root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 10))
        
        # =================================================================
        # ENTRY - Champs de texte modernes
        # =================================================================
        style.configure(
            "TEntry",
            fieldbackground=colors['white'],
            foreground=colors['text_primary'],
            bordercolor=colors['border'],
            lightcolor=colors['border_light'],
            insertcolor=colors['accent_primary'],
            padding=[8, 6],
            font=("Segoe UI", 10)
        )
        
        style.map(
            "TEntry",
            bordercolor=[
                ("focus", colors['accent_primary'])
            ],
            lightcolor=[
                ("focus", colors['accent_light'])
            ]
        )
        
        # =================================================================
        # SCALE (Sliders) - Style moderne
        # =================================================================
        style.configure(
            "TScale",
            background=colors['bg_primary'],
            troughcolor=colors['border_light'],
            sliderlength=20,
            sliderthickness=20
        )
        
        style.configure(
            "Horizontal.TScale",
            background=colors['bg_primary'],
            troughcolor=colors['bg_accent'],
            sliderrelief="flat"
        )
        
        style.map(
            "Horizontal.TScale",
            background=[
                ("active", colors['accent_hover'])
            ]
        )
        
        # =================================================================
        # SCROLLBAR - Style minimaliste moderne
        # =================================================================
        style.configure(
            "Vertical.TScrollbar",
            background=colors['bg_accent'],
            troughcolor=colors['bg_secondary'],
            bordercolor=colors['bg_secondary'],
            arrowcolor=colors['accent_primary'],
            relief="flat",
            width=12
        )
        
        style.map(
            "Vertical.TScrollbar",
            background=[
                ("active", colors['accent_light']),
                ("pressed", colors['accent_primary'])
            ]
        )
        
        style.configure(
            "Horizontal.TScrollbar",
            background=colors['bg_accent'],
            troughcolor=colors['bg_secondary'],
            bordercolor=colors['bg_secondary'],
            arrowcolor=colors['accent_primary'],
            relief="flat",
            width=12
        )
        
        style.map(
            "Horizontal.TScrollbar",
            background=[
                ("active", colors['accent_light']),
                ("pressed", colors['accent_primary'])
            ]
        )
        
        # =================================================================
        # TREEVIEW - Table moderne
        # =================================================================
        style.configure(
            "Treeview",
            background=colors['white'],
            foreground=colors['text_primary'],
            fieldbackground=colors['white'],
            rowheight=28,
            font=("Segoe UI", 10)
        )
        
        style.configure(
            "Treeview.Heading",
            background=colors['accent_primary'],
            foreground=colors['white'],
            font=("Segoe UI", 10, "bold"),
            padding=[8, 6],
            relief="flat"
        )
        
        style.map(
            "Treeview.Heading",
            background=[
                ("active", colors['accent_dark'])
            ]
        )
        
        style.map(
            "Treeview",
            background=[
                ("selected", colors['accent_light'])
            ],
            foreground=[
                ("selected", colors['white'])
            ]
        )
        
        # =================================================================
        # SEPARATOR
        # =================================================================
        style.configure(
            "TSeparator",
            background=colors['border_light']
        )
        
        # =================================================================
        # PROGRESSBAR
        # =================================================================
        style.configure(
            "TProgressbar",
            background=colors['accent_primary'],
            troughcolor=colors['bg_accent'],
            bordercolor=colors['border'],
            lightcolor=colors['accent_light'],
            darkcolor=colors['accent_dark']
        )
        
        # =================================================================
        # CHECKBUTTON & RADIOBUTTON
        # =================================================================
        style.configure(
            "TCheckbutton",
            background=colors['bg_primary'],
            foreground=colors['text_primary'],
            font=("Segoe UI", 10),
            focuscolor=""
        )
        
        style.map(
            "TCheckbutton",
            background=[
                ("active", colors['bg_secondary'])
            ],
            indicatorcolor=[
                ("selected", colors['accent_primary'])
            ]
        )
        
        style.configure(
            "TRadiobutton",
            background=colors['bg_primary'],
            foreground=colors['text_primary'],
            font=("Segoe UI", 10),
            focuscolor=""
        )
        
        style.map(
            "TRadiobutton",
            background=[
                ("active", colors['bg_secondary'])
            ],
            indicatorcolor=[
                ("selected", colors['accent_primary'])
            ]
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
        
        # Columns without Date and Time
        columns = ('ID', 'CO', 'NO2', 'Temperature', 'Humidity')
        self.data_tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=15)
        
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
        
        # Buttons for CRUD operations
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

        # Conteneur pour le tableau et la visualisation (même taille)
        content_frame = ttk.Frame(tab)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Configurer le grid pour partager l'espace équitablement
        content_frame.grid_columnconfigure(0, weight=1, uniform="equal")
        content_frame.grid_columnconfigure(1, weight=1, uniform="equal")
        content_frame.grid_rowconfigure(0, weight=1)

        # Frame pour le tableau filtré
        filtered_frame = ttk.LabelFrame(content_frame, text="Filtered Data", padding=10)
        filtered_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        # Créer le Treeview avec les mêmes colonnes que le tableau principal
        filter_columns = ('ID', 'CO', 'NO2', 'Temperature', 'Humidity')
        self.filtered_tree = ttk.Treeview(filtered_frame, columns=filter_columns, show='headings', height=15)
        for col in filter_columns:
            self.filtered_tree.heading(col, text=col)
            self.filtered_tree.column(col, width=80)

        # Scrollbars
        vsb_f = ttk.Scrollbar(filtered_frame, orient=tk.VERTICAL, command=self.filtered_tree.yview)
        hsb_f = ttk.Scrollbar(filtered_frame, orient=tk.HORIZONTAL, command=self.filtered_tree.xview)
        self.filtered_tree.configure(yscrollcommand=vsb_f.set, xscrollcommand=hsb_f.set)

        self.filtered_tree.grid(row=0, column=0, sticky='nsew')
        vsb_f.grid(row=0, column=1, sticky='ns')
        hsb_f.grid(row=1, column=0, sticky='ew')

        filtered_frame.grid_rowconfigure(0, weight=1)
        filtered_frame.grid_columnconfigure(0, weight=1)

        # Frame pour la visualisation (même taille que le tableau)
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
        self.notebook.add(tab, text="Spectrum Analysis")
        
        left_frame = ttk.LabelFrame(tab, text="Controls", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # === SECTION 1: Variable Selection ===
        ttk.Label(left_frame, text="1. Select Variable", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(left_frame, text="Choose the time series to analyze:").pack(anchor=tk.W, pady=(2, 5))
        self.spectral_var = ttk.Combobox(left_frame, values=self.display_columns, state='readonly', width=20)
        self.spectral_var.set(self.display_columns[0])
        self.spectral_var.pack(fill=tk.X, pady=(0, 10))
        
        # === SECTION 2: FFT Analysis ===
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(left_frame, text="2. FFT Analysis", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        ttk.Button(left_frame, text="▶ Run FFT Analysis", command=self.run_spectral_analysis).pack(fill=tk.X, pady=5)
        
        # === SECTION 3: Frequency Filters ===
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="3. Apply Filter (optional)", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(left_frame, text="Filter the frequency signal:").pack(anchor=tk.W, pady=(2, 5))
        
        # Filter type
        self.spectral_filter_type = ttk.Combobox(left_frame, values=['Low-pass', 'High-pass', 'Band-pass'], state='readonly', width=20)
        self.spectral_filter_type.set('Low-pass')
        self.spectral_filter_type.pack(fill=tk.X, pady=5)
        self.spectral_filter_type.bind('<<ComboboxSelected>>', self._update_filter_fields)
        
        # Filter parameters frame
        self.filter_params_frame = ttk.Frame(left_frame)
        self.filter_params_frame.pack(fill=tk.X, pady=5)
        
        # Cutoff frequency (for low-pass and high-pass)
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
        
        ttk.Button(left_frame, text="Apply Filter", command=self.apply_spectral_filter).pack(fill=tk.X, pady=5)
        
        # === Results Section ===
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Results", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.spectral_results = scrolledtext.ScrolledText(left_frame, height=8, width=25, font=('Consolas', 9))
        self.spectral_results.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Save to Database", command=self.save_spectral_results).pack(fill=tk.X, pady=10)
        
        # === Right Frame - Spectrum Plot ===
        right_frame = ttk.LabelFrame(tab, text="Power Spectrum", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.spectral_fig = Figure(figsize=(8, 6), dpi=100)
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_fig, right_frame)
        self.spectral_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.spectral_canvas, toolbar_frame)
    
    def _update_filter_fields(self, event=None):
        """Update filter parameter fields based on selected filter type"""
        filter_type = self.spectral_filter_type.get()
        
        # Hide all frames first
        self.cutoff_frame.pack_forget()
        self.bandpass_frame.pack_forget()
        
        # Show appropriate frame
        if filter_type in ['Low-pass', 'High-pass']:
            self.cutoff_frame.pack(fill=tk.X)
        elif filter_type == 'Band-pass':
            self.bandpass_frame.pack(fill=tk.X)
    
    #ONGLET 5: TRAITEMENT D'IMAGES
    
    def create_image_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Image Processing")

        # Panneau gauche sans scroll - utilisation de grid pour un layout compact
        left_frame = ttk.LabelFrame(tab, text="Processing", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # --- Section: Load Image ---
        ttk.Button(left_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Load Image from DB", command=self.load_image_from_db).pack(fill=tk.X, pady=3)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        # --- Section: Processing avec Dropdown ---
        ttk.Label(left_frame, text="Processing:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        # Mapping des noms affichés vers les clés de traitement
        self.processing_options = {
            'Grayscale Conversion': 'grayscale',
            'Gaussian Blur': 'blur',
            'Canny Edge Detection': 'canny',
            'Otsu Thresholding': 'otsu',
            'Adaptive Thresholding': 'adaptive',
            'Sobel Filter': 'sobel'
        }
        
        # Dropdown menu pour sélection du traitement
        self.processing_combo = ttk.Combobox(
            left_frame, 
            values=list(self.processing_options.keys()),
            state='readonly',
            width=22
        )
        self.processing_combo.set('Grayscale Conversion')
        self.processing_combo.pack(fill=tk.X, pady=5)
        
        # Bouton Apply pour appliquer le traitement sélectionné
        ttk.Button(left_frame, text="Apply Processing", command=self.apply_selected_processing).pack(fill=tk.X, pady=3)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        # --- Section: Parameters ---
        ttk.Label(left_frame, text="Parameters:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        # Paramètre flou (KERNEL IMPAIR UNIQUEMENT)
        ttk.Label(left_frame, text="Blur Kernel:").pack(anchor=tk.W, pady=(5, 0))
        
        kernel_frame = ttk.Frame(left_frame)
        kernel_frame.pack(fill=tk.X)
        
        self.blur_kernel = ttk.Scale(
            kernel_frame,
            from_=3,
            to=21,
            orient=tk.HORIZONTAL,
            command=self.update_blur_kernel
        )
        self.blur_kernel.set(5)
        self.current_blur_kernel = 5
        self.blur_kernel.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.blur_kernel_label = ttk.Label(kernel_frame, text="5", width=3)
        self.blur_kernel_label.pack(side=tk.RIGHT, padx=5)

        # Seuils Canny
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
        
        # --- Section: Actions ---
        ttk.Button(left_frame, text="Reset", command=self.reset_image).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Save", command=self.save_processed_image).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="Store Metadata", command=self.store_processed_image_metadata).pack(fill=tk.X, pady=3)
        
        # --- Zone d'affichage à droite ---
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
        """Applique le traitement sélectionné dans le dropdown menu"""
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
            # Afficher tous les enregistrements (sans date et time)
            for idx, row in self.data.iterrows():
                values = (
                    row.get('id', idx),
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
        """Add a new row to the database"""
        # Create a dialog window for adding data
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Row")
        dialog.geometry("300x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        fields = ['CO', 'NO2', 'Temperature', 'Humidity']
        entries = {}
        
        for i, field in enumerate(fields):
            ttk.Label(dialog, text=f"{field}:").grid(row=i, column=0, padx=10, pady=5, sticky='w')
            entry = ttk.Entry(dialog)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky='ew')
            entries[field] = entry
        
        def save_new_row():
            try:
                self.db.connect()
                co = float(entries['CO'].get()) if entries['CO'].get() else None
                no2 = float(entries['NO2'].get()) if entries['NO2'].get() else None
                temp = float(entries['Temperature'].get()) if entries['Temperature'].get() else None
                hum = float(entries['Humidity'].get()) if entries['Humidity'].get() else None
                
                self.db.insert_measurement(
                    date="N/A", time="N/A",
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
        """Edit the selected row"""
        selected = self.data_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a row to edit")
            return
        
        item = self.data_tree.item(selected[0])
        values = item['values']
        record_id = values[0]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Row")
        dialog.geometry("300x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        fields = ['CO', 'NO2', 'Temperature', 'Humidity']
        entries = {}
        
        for i, field in enumerate(fields):
            ttk.Label(dialog, text=f"{field}:").grid(row=i, column=0, padx=10, pady=5, sticky='w')
            entry = ttk.Entry(dialog)
            entry.insert(0, str(values[i+1]) if values[i+1] else '')
            entry.grid(row=i, column=1, padx=10, pady=5, sticky='ew')
            entries[field] = entry
        
        def save_edit():
            try:
                self.db.connect()
                updates = {}
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
        """Delete the selected row"""
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
        ax = self.filter_fig.add_subplot(111)
        
        var_selected = self.filter_var.get()         
        df_column = self.COLUMN_MAP[var_selected]  
        
        # Create a working copy of data
        filtered_data = self.data.copy()
        original = filtered_data[df_column].dropna().values[:1000]

        
        #Appliquer le filtre
        if filter_type == 'Moving Average':
            filtered = pd.Series(original).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            title = f"Moving Average (window={window})"
        elif filter_type == 'Threshold Filter':
            min_val = float(self.threshold_min.get())
            max_val = float(self.threshold_max.get())
            # Filter rows based on threshold
            mask = (filtered_data[df_column] >= min_val) & (filtered_data[df_column] <= max_val)
            filtered_data = filtered_data[mask]
            filtered = np.clip(original, min_val, max_val)
            title = f"Thresholding [{min_val}, {max_val}]"
        else:  # Outliers
            q1, q3 = np.percentile(original, [25, 75])
            iqr = q3 - q1
            filtered = np.clip(original, q1 - 1.5*iqr, q3 + 1.5*iqr)
            title = "Remove Outliers (IQR)"
        
        # Update the filtered_tree with same columns as main table
        for item in self.filtered_tree.get_children():
            self.filtered_tree.delete(item)

        # Show filtered data with all columns (same format as main table)
        for idx, row in filtered_data.head(1000).iterrows():
            values = (
                row.get('id', idx),
                f"{row.get('co_gt', 0):.2f}" if pd.notna(row.get('co_gt')) else '',
                f"{row.get('no2_gt', 0):.2f}" if pd.notna(row.get('no2_gt')) else '',
                f"{row.get('temperature', 0):.1f}" if pd.notna(row.get('temperature')) else '',
                f"{row.get('humidity', 0):.1f}" if pd.notna(row.get('humidity')) else ''
            )
            self.filtered_tree.insert('', tk.END, values=values)

        
        # Simple graph in reduced space
        ax.plot(original[:200], 'b-', linewidth=0.5, alpha=0.5, label='Original')
        ax.plot(filtered[:200], 'r-', linewidth=1, label='Filtered')
        ax.set_title(f'{var}: {title}', fontsize=8, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        
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
        
        # Show complete heatmap without mask (including all correlation values)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
        
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
        
        # Use display names instead of column names
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
        """Run FFT analysis and display power spectrum"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        try:
            var_selected = self.spectral_var.get()
            var_fft = self.COLUMN_MAP[var_selected]
            signal = self.data[var_fft].dropna().values
            
            if len(signal) < 10:
                messagebox.showwarning("Warning", "Not enough data points for analysis")
                return
            
            # Remove mean (DC component)
            signal = signal - np.mean(signal)
            
            self.spectral_fig.clear()
            
            # Create 2 subplots: time signal + power spectrum
            ax1 = self.spectral_fig.add_subplot(211)
            ax2 = self.spectral_fig.add_subplot(212)
            
            # Plot temporal signal (first 500 points)
            display_len = min(500, len(signal))
            ax1.plot(signal[:display_len], 'b-', linewidth=0.8)
            ax1.set_title(f'Time Signal: {var_selected}', fontweight='bold')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            
            # Compute Power Spectrum using Welch method
            nperseg = min(256, len(signal)//4)
            if nperseg < 4:
                nperseg = len(signal)
                
            frequencies, power = welch(signal, fs=1.0, nperseg=nperseg)
            
            # Plot Power Spectrum
            ax2.semilogy(frequencies, power, 'r-', linewidth=1)
            ax2.axvline(x=1/24, color='green', linestyle='--', alpha=0.7, label='Daily (24h)')
            ax2.axvline(x=1/168, color='orange', linestyle='--', alpha=0.7, label='Weekly (168h)')
            ax2.set_title(f'Power Spectrum: {var_selected}', fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Power Spectral Density')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3, which='both')
            
            # Find and display dominant frequencies
            peak_idx = np.argsort(power)[-5:][::-1]
            results = f"FFT Analysis: {var_selected}\n"
            results += "-" * 25 + "\n"
            results += "Dominant Frequencies:\n\n"
            for idx in peak_idx:
                f = frequencies[idx]
                if f > 0:
                    period = 1/f
                    results += f"• {f:.5f} Hz\n"
                    results += f"  Period: {period:.1f}h ({period/24:.1f} days)\n\n"
            
            self.spectral_results.delete(1.0, tk.END)
            self.spectral_results.insert(tk.END, results)
            
            self.spectral_fig.tight_layout()
            self.spectral_canvas.draw()
            self.log(f"FFT Analysis completed: {var_selected}")
            
        except Exception as e:
            self.log(f"Error in spectral analysis: {e}")
            messagebox.showerror("Error", f"Spectral analysis failed: {e}")
    
    def apply_spectral_filter(self):
        """Apply frequency filter and show effect on power spectrum"""
        from scipy.signal import butter, sosfilt
        
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded. Run FFT Analysis first.")
            return
        
        filter_type = self.spectral_filter_type.get()
        
        try:
            var_selected = self.spectral_var.get()
            var_fft = self.COLUMN_MAP[var_selected]
            signal = self.data[var_fft].dropna().values.copy()
            
            if len(signal) < 20:
                messagebox.showwarning("Warning", "Not enough data points for filtering")
                return
            
            # Remove mean
            signal = signal - np.mean(signal)
            
            fs = 1.0  # Sampling frequency (1 sample per hour)
            nyq = 0.5 * fs  # Nyquist frequency = 0.5 Hz
            order = 3  # Filter order
            
            if filter_type == 'Low-pass':
                cutoff = float(self.cutoff_freq.get())
                if cutoff <= 0 or cutoff >= nyq:
                    messagebox.showwarning("Warning", f"Cutoff must be between 0 and {nyq} Hz")
                    return
                sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
                filtered_signal = sosfilt(sos, signal)
                title = f"Low-pass (cutoff={cutoff} Hz)"
                
            elif filter_type == 'High-pass':
                cutoff = float(self.cutoff_freq.get())
                if cutoff <= 0 or cutoff >= nyq:
                    messagebox.showwarning("Warning", f"Cutoff must be between 0 and {nyq} Hz")
                    return
                sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
                filtered_signal = sosfilt(sos, signal)
                title = f"High-pass (cutoff={cutoff} Hz)"
                
            elif filter_type == 'Band-pass':
                low_cut = float(self.low_cutoff.get())
                high_cut = float(self.high_cutoff.get())
                if low_cut <= 0 or high_cut >= nyq or low_cut >= high_cut:
                    messagebox.showwarning("Warning", f"Frequencies must satisfy: 0 < low < high < {nyq}")
                    return
                sos = butter(order, [low_cut, high_cut], btype='band', fs=fs, output='sos')
                filtered_signal = sosfilt(sos, signal)
                title = f"Band-pass ({low_cut}-{high_cut} Hz)"
            else:
                return
            
            # Compute power spectra for both signals
            nperseg = min(256, len(signal)//4)
            if nperseg < 4:
                nperseg = len(signal)
            
            freq_orig, power_orig = welch(signal, fs=fs, nperseg=nperseg)
            freq_filt, power_filt = welch(filtered_signal, fs=fs, nperseg=nperseg)
            
            # Plot results
            self.spectral_fig.clear()
            ax1 = self.spectral_fig.add_subplot(211)
            ax2 = self.spectral_fig.add_subplot(212)
            
            # Time domain comparison
            display_len = min(500, len(signal))
            ax1.plot(signal[:display_len], 'b-', linewidth=0.5, alpha=0.5, label='Original')
            ax1.plot(filtered_signal[:display_len], 'r-', linewidth=1, label='Filtered')
            ax1.set_title(f'Time Signal: {title}', fontweight='bold')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Value')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Frequency domain comparison (Power Spectrum)
            ax2.semilogy(freq_orig, power_orig, 'b-', linewidth=0.8, alpha=0.5, label='Original')
            ax2.semilogy(freq_filt, power_filt, 'r-', linewidth=1.5, label='Filtered')
            ax2.set_title(f'Power Spectrum: {title}', fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Power Spectral Density')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3, which='both')
            
            self.spectral_fig.tight_layout()
            self.spectral_canvas.draw()
            
            # Update results text
            self.spectral_results.delete(1.0, tk.END)
            self.spectral_results.insert(tk.END, f"Filter Applied:\n{title}\n\n")
            self.spectral_results.insert(tk.END, f"Original std: {np.std(signal):.2f}\n")
            self.spectral_results.insert(tk.END, f"Filtered std: {np.std(filtered_signal):.2f}\n\n")
            self.spectral_results.insert(tk.END, "Tip: Compare blue (original)\nand red (filtered) curves\nto see filter effect.")
            
            self.log(f"Filter Applied: {title}")
            
        except ValueError as ve:
            self.log(f"Value error: {ve}")
            messagebox.showerror("Error", f"Invalid frequency value: {ve}")
        except Exception as e:
            self.log(f"Error applying filter: {e}")
            messagebox.showerror("Error", f"Failed to apply filter: {e}")
    
    def save_spectral_results(self):
        #Sauvegarde les résultats spectraux
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
