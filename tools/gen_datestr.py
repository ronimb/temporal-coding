#  This is used for easily generating a date time string, either for date only or for date and time
from datetime import datetime as dt
def gen_datestr(with_time=True, spaces=True):
    """

    :param with_time: whether to include time of day or not
    :param spaces: whether to include spaces or not (used for folder naming)
    """
    if with_time:
        datestr = dt.now().strftime('%b_%d %H:%M:%S')
    else:
        datestr = dt.now().strftime('%b_%d')

    if not spaces:
        datestr.replace(' ', '_')
    return datestr
