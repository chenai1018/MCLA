from MCLA import MCLA
from util.conf import ModelConf
import time

if __name__ == '__main__':
    s = time.time()

    # conf = ModelConf('./conf/' + 'MCLA_douban' + '.conf')  # For Douban Music Dataset
    conf = ModelConf('./conf/' + 'MCLA_Epinions' + '.conf')  # For Epinions Dataset
    # conf = ModelConf('./conf/' + 'MCLA_yelp' + '.conf')  # For Yelp Dataset

    rec = MCLA(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
