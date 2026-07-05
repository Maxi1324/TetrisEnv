# tetris-env-cuda

CUDA vectorized Tetris environment exposed as PyTorch custom operators.

## Install

From GitHub:

```powershell
pip install git+https://github.com/Maxi1324/TetrisEnv.git
```

Run from the repository root:

```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%PATH%
pip install -e .
```

On Windows, run this from a Visual Studio Developer Command Prompt or load
`vcvars64.bat` before installing.

## Usage

```python
import torch
import tetris_env

envs = 1024
tetris_env.start(envs, envs_per_thread=1)

actions = torch.zeros(envs, device="cuda", dtype=torch.uint32)
observations = torch.empty((envs, 18), device="cuda", dtype=torch.float32)
rewards = torch.empty(envs, device="cuda", dtype=torch.float32)
done = torch.empty(envs, device="cuda", dtype=torch.bool)

stats = tetris_env.step(actions, observations, rewards, done, image_observation=False)
torch.cuda.synchronize()

avg_episode_length = stats[0]
avg_rows_cleared = stats[1]
```
