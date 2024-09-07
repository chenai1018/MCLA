from data.loader import FileIO


class MCLA(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.rating_split_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'])
        self.test_data = FileIO.load_data_set(config['test.set'])

        self.kwargs = {}

        if config.contain('rating.split'):
            rating_split_data = FileIO.load_rating_split_data(self.config['training.set'], self.config['rating.split'])
            self.kwargs['rating.split'] = rating_split_data

        if config.contain('friend.data'):
            friend_data = FileIO.load_social_data(self.config['friend.data'])
            self.kwargs['friend.data'] = friend_data

        if config.contain('group.data'):
            group_data = FileIO.load_social_data(self.config['group.data'])
            self.kwargs['group.data'] = group_data

        print('Reading data and preprocessing...')

    def execute(self):
        import_str = 'from model.'+ self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        eval(recommender).execute()