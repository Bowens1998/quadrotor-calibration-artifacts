"""All original 94 development configurations under the shared altitude intervention."""
import icra_mpc_matrix as parent
from icra_vertical_authority import command_with_cap
BASE=parent.ROOT/'runs/icra_mpc_authority_matrix_20260908'
original_rollout=parent.rollout

def rollout(*args,**kwargs):
    calls=0
    def command(pos,ref,velocity):
        nonlocal calls
        calls+=1
        return command_with_cap(pos,ref,velocity,.5 if calls<=100 else 1.5)
    parent.nominal_command=command
    result=original_rollout(*args,**kwargs)
    assert calls==600
    return result

if __name__=='__main__':
    parent.BASE=BASE
    parent.rollout=rollout
    parent.main()
