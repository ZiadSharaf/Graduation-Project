from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo, showerror
from tkinter.simpledialog import askinteger
from modeling.modules.constants import *

# all features
SELECTION = ['wc', 'fm', 'tg', 'sex', 'hdl', 'ggt', 'height', 'ast', 'hba1c', 'fpg', 'tc', 'sbp', 'age']

# updated feature sets
NOMS = [col for col in NOMS if col in SELECTION]
ORDS = [col for col in ORDS if col in SELECTION]
CATS = NOMS + ORDS
DISCS = [col for col in DISCS if col in SELECTION]
CONTS = [col for col in CONTS if col in SELECTION]
NUMS = DISCS + CONTS

class App(Tk):
    def __init__(self):
        super().__init__()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(26, weight=1)
        self.title("NAFLD Prediction")
        self.geometry('600x610')

        self.columns_widgets = {}
        self.columns_vars = {}

        # Main Frame

        self.bg = PhotoImage(file="bg.png")
        Label(self, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)

        Label(self, text=TEXT_DICT['sex'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5, pady=(5, 0))
        self.columns_widgets['sex'] = ttk.Combobox(self, values=CAT_DICT['sex'], font=("TkDefaultFont", 10))
        self.columns_widgets['sex'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['age'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['age'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['age'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['height'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['height'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['height'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['fm'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['fm'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['fm'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['wc'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['wc'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['wc'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['ast'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['ast'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['ast'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['ggt'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['ggt'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['ggt'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['hdl'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['hdl'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['hdl'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['tc'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['tc'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['tc'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['tg'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['tg'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['tg'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['fpg'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['fpg'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['fpg'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['hba1c'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['hba1c'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['hba1c'].grid(sticky="w", padx=5)

        Label(self, text=TEXT_DICT['sbp'], font=("TkDefaultFont", 10)).grid(sticky='w', padx=5)
        self.columns_widgets['sbp'] = ttk.Spinbox(self, to=float("inf"), font=("TkDefaultFont", 10))
        self.columns_widgets['sbp'].grid(sticky="w", padx=5)

        Button(self, text="Predict", command=self.classify, bg='#aaaaaa', font=("TkDefaultFont", 10)).grid(sticky="wes", padx=5)

        # Load Ensemble

        import sys, os, pickle
        sys.path.append(os.path.join(os.getcwd(), "modeling"))
        with open("models/ensemble_selection.pkl", 'rb') as f:
            self.ensemble = pickle.load(f)

    def get_data(self):
        import pandas as pd
        sample_df = pd.DataFrame()
        try:
            for col in SELECTION:
                widget = self.columns_widgets[col]
                if col in CAT_DICT.keys():
                    sel = widget.get()
                    sample_df[col]=[CAT_DICT[col].index(sel)]
                else:
                    val = widget.get()
                    sample_df[col]=[val]

            sample_df[DISCS] = sample_df[DISCS].astype('int')
            sample_df[CONTS] = sample_df[CONTS].astype('float')
            sample_df[NOMS] = sample_df[NOMS].astype(pd.CategoricalDtype())
            sample_df[ORDS] = sample_df[ORDS].astype(pd.CategoricalDtype(ordered=True))
        except ValueError:
            showerror("Invalid Input", "One or more variables have invalid input.")
            return
        for col in NUMS:
            if (sample_df[col] < 0).any():
                showerror("Invalid Input", f"Value of {TEXT_DICT[col]} is negative")
                return
        return sample_df

    def classify(self):
        sample_df = self.get_data()
        if sample_df is None: return

        probs = self.ensemble.predict_proba(sample_df)[0]
        showinfo("Result", f"Probability of having NAFLD is {round(probs[1], 4)}")

        # FIB-4 score calculation

        plt_count = askinteger("FIB-4 Score", "Enter platelet count to calculate FIB-4 score", parent=self)
        alt = askinteger("FIB-4 Score", "Enter ALT to calculate FIB-4 score", parent=self)
        fib4 = (sample_df['age'][0] * sample_df['ast'][0]) / (plt_count * pow(alt, 0.5))
        if fib4 < 1.45:
            showinfo("FIB-4 Score", f"Score is {fib4}, minimal to no fibrosis")
        elif fib4 >= 1.45 and fib4 < 3.4:
            showinfo("FIB-4 Score", f"Score is {fib4}, significant fibrosis")
        else:
            showinfo("FIB-4 Score", f"Score is {fib4}, advanced fibrosis or cirrhosis")

app = App()
app.mainloop()
