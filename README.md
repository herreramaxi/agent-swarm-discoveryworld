# DiscoveryWorld video export

Run the multi-agent notebook to generate shared-map frames. The notebook stores
them in the directory shown by `print(api.FRAME_DIR)`.

From the project root, pass that directory to the exporter:

```powershell
uv run python -m a4ai_project.video video/frames-thread1794573398746427500
```

The exporter uses the project's bundled FFmpeg and saves `shared-world.mp4`
inside the run directory at 2 frames per second. Use `--fps 4` to speed up
playback or `--output video/both-agents.mp4` to choose another destination.
