from IPython.display import display, clear_output
from ipywidgets import widgets


class Learn:
    def __init__(self, max_count=100):
        self.count = 0
        self.max_count = max_count

    def on_button_clicked(self, b):
        self.choices[b.description]()
        self.container.close()
        clear_output()
        if self.count < self.max_count:
            self.count += 1
            self.run()

    def run(self):
        if self.count > self.max_count:
            return
        buttons = [widgets.Button(description=n) for n in self.choices]
        self.container = widgets.HBox(children=buttons)
        self.plot()
        display(self.container)
        for button in buttons:
            button.on_click(self.on_button_clicked)
