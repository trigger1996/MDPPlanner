#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from subprocess import check_output
import re
import shutil
from User.vis2  import print_c

def ltl_convert(task, is_display=True):
    #
    # https://www.ltl2dstar.de/docs/ltl2dstar.html#:~:text=ltl2dstar%20is%20designed%20to%20use%20an%20external%20tool%20to%20convert
    # LTL是有两个格式的, 一个ltl2dstar notation, 一个spin notation
    # 后者是常用的, 要从后者转到前者ltl2dstar才能用
    if shutil.which('ltlfilt'):
        cmd_ltl_convert = 'ltlfilt -l -f \'%s\'' % (task, )
        ltl_converted = str(check_output(cmd_ltl_convert, shell=True))
        ltl_converted = ltl_converted[2 : len(ltl_converted) - 3]           # tested
    else:
        # Prefix/LBT form used by the 0506 experiments.  This fallback keeps
        # the project runnable on Windows where Spot/ltlfilt is unavailable.
        match = re.fullmatch(
            r'GF\s*\(gather\s*->\s*\(!gather\s+U\s+drop\)\)\s*&\s*GF\s+([A-Za-z_][A-Za-z0-9_]*)',
            task.strip(),
        )
        if not match:
            raise RuntimeError('ltlfilt is unavailable and no built-in conversion matches: ' + task)
        ltl_converted = '& G F | ! gather U ! gather drop G F ' + match.group(1)
    if is_display:
        print_c('converted ltl: ' + ltl_converted)

    return ltl_converted
