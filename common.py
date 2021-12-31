from time import time

def timer(func):
    def wrapper_function(*args, **kwargs):
        t0 = time()
        print(f'\n Running {func.__name__}')
        out = func(*args,  **kwargs)
        seconds = int(time()-t0)
        minutes = round(seconds/60, 1)
        print(f'\n{func.__name__} time taken: {seconds}s={minutes}mins')
        return out
    return wrapper_function

def progress(i, N, title=''):
    if int(i/N*100)!=int((i-1)/N*100):
        print(f'{title}: {int(i/N*100)}%', end='\r')
