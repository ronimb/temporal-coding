import sys
import subprocess
# %%
if __name__ == '__main__':
    # print(__file__)
    # sys.path.extend('/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/')
    print(sys.argv[1])
    exec(open(sys.argv[1]).read())