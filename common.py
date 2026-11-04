import os
import pickle
import sys
import matplotlib
import numpy as np
import gymnasium as gym

matplotlib.use('Agg')
import matplotlib.pyplot as plt

RENDER_TRAINING=False

HERE=os.path.dirname(os.path.abspath(__file__))

def here(name):
    # Join the script folder path with the given filename.
    return os.path.join(HERE,name)

_FORMAT='qtable-v1'

def save_model(q,name):
    q=np.ascontiguousarray(q)
    payload={
        'format':_FORMAT,
        'dtype':q.dtype.str,
        'shape':q.shape,  
        'data':q.tobytes(),
    }

    path=here(name)
    tmp=path+'.tmp'


    with open(tmp,'wb') as f:
        pickle.dump(payload,f,protocol=4)


    os.replace(tmp,path)

def load_model(name):
    path=here(name)

    script=os.path.basename(sys.argv[0])

    if not os.path.exists(path):
        raise SystemExit(
            f"No saved model found: {path}\n"
            f"Train one first: python {script} train"
        )

    try:
        with open(path,'rb') as f:
            obj=pickle.load(f)

    except (ModuleNotFoundError,AttributeError,ImportError,
            pickle.UnpicklingError,EOFError) as e:
        raise SystemExit(
            f"Can't read {path}\n"
            f"  reason: {type(e).__name__}: {e}\n"
            f"Usually the file was written using a different NumPy/Python version "
            f"or was damaged while saving.\n"
            f"Delete it and retrain: python {script} train"
        )

    if isinstance(obj,dict) and obj.get('format')==_FORMAT:

        arr=np.frombuffer(obj['data'],dtype=np.dtype(obj['dtype']))

        return arr.reshape(obj['shape'])

    return obj


def make_env(env_id,render,**kwargs):
    try:
        return gym.make(
            env_id,
            render_mode='human' if render else None,
            **kwargs
        )

    except gym.error.Error as e:
        raise SystemExit(
            f"gymnasium {gym.__version__} can't create '{env_id}': {e}\n"
            f"Fix: pip install -U \"gymnasium[classic-control,toy-text]\""
        )

    except ImportError as e:
        raise SystemExit(
            f"Missing dependency for '{env_id}': {e}\n"
            f"Fix: pip install \"gymnasium[classic-control,toy-text]\""
        )


def rolling(values,window=100,how='mean'):
    v=np.asarray(values,dtype=float)
    c=np.concatenate(([0.0],np.cumsum(v)))

    end=np.arange(1,len(v)+1)
    start=np.maximum(end-window,0)
    total=c[end]-c[start]

    return total/(end-start) if how=='mean' else total

def save_plot(values,name,how='mean'):
    if len(values)==0:
        return

    fig=plt.figure()
    plt.plot(rolling(values,how=how))
    plt.xlabel('episode')
    plt.ylabel(f'{how} reward over last 100 episodes')
    fig.savefig(here(name))

    plt.close(fig)

def save_progress(q,pkl,png,rewards_per_episode,how='mean'):
    n=len(rewards_per_episode)

    if n==0:
        print('No episode finished - nothing saved (old model, if any, is untouched).')
        return

    save_model(q,pkl)
    save_plot(rewards_per_episode,png,how=how)

    print(f'Saved {pkl} and {png} after {n} episodes.')


def run_cli(train_fn,play_fn,pkl):

    args=[a.lower() for a in sys.argv[1:]]
    render_training=RENDER_TRAINING or '--render' in args or 'render' in args

    have_model=os.path.exists(here(pkl))
    do_train='train' in args or (not have_model and 'play' not in args)

    try:
        if do_train:
            print(
                f"Training... window while training: "
                f"{'ON (slow)' if render_training else 'OFF (fast)'}"
                f"  |  Ctrl+C stops and saves progress"
            )

            train_fn(render_training)

        print("Playing the learned policy (window). Ctrl+C to quit.")
        play_fn()

    except KeyboardInterrupt:
    
        print("\nStopped by Ctrl+C."+(" Progress was saved." if do_train else ""))
        sys.exit(130)