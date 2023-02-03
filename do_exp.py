import os
from subprocess import call
import sys



def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)


def run(cfg_file, scriptName):

    flags = '--%s %s' % ('config', cfg_file)
    call('python ' + scriptName + '.py' +' %s' % flags, shell=True)
    

if __name__ == "__main__":

    run(sys.argv[1], sys.argv[2])
            
            

