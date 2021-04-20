
class Edge(Learn):
    def __init__(self, file_lst=file_lst, input_parameters=None, **arg):
        if input_parameters is None:
            input_parameters = {
            'sigma': 30,
            'low': 10,
            'high': 10
        }
        self.parameters = input_parameters
        self.new_parameters = self.parameters.copy()
        self.file_list = file_lst
        self.data = []
        self.choices = {
            'perfect': self.perfect,
            'better': self.better,
            'bad': self.bad,
            'skip': self.skip
        }
        self.analyse = None
        self.skip()
        super().__init__(**arg)

    def perfect(self):
        self.parameters = self.new_parameters.copy()
        self.skip()

    def skip(self):
        if self.analyse is not None:
            self.data.append([
                self.analyse.get_base_color()/255,
                self.new_parameters.copy()
            ])
        self.file_name = np.random.choice(self.file_list)
        self.analyse = Analyse(self.file_name, initialize=False)

    def better(self):
        self.parameters = self.new_parameters.copy()
        self.count -= 1

    def bad(self):
        self.count -= 1

    def plot(self):
        img_color = self.analyse.get_image()
        img = self.analyse.get_image(True)
        base = self.analyse.get_base_color()/255
        for k,v in self.parameters.items():
            self.new_parameters[k] = np.absolute(np.random.randn()/5+1)*v
        _, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img, cmap='Greys')
        old_map = 1*(get_edge(self.parameters, img, base))
        new_map = 2*(get_edge(self.new_parameters, img, base))
        total_map = old_map+new_map
        for ii, n in enumerate(['old', 'new', 'both']):
            x = np.where(total_map==ii+1)
            ax[0].scatter(x[1], x[0], marker='.', label=n, s=0.5)
        ax[0].legend()
#         for i, param in enumerate([self.parameters, self.new_parameters]):
#             edge = get_edge(param, img, base)
#             ax[i].imshow(edge)
#         ax[1].imshow(get_edge(self.new_parameters, img, base))
        ax[1].imshow(img_color)
