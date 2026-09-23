# /// script
# dependencies = [
#   "cuml-cu12", # or cuml-cu11 depending on your CUDA version
#   "cupy-cuda12x",
#   "numpy",
#   "plotext",
# ]
# ///

import sys
import curses
import cupy as cp
import plotext as plt
from cuml.neighbors import NearestNeighbors
from cuml.manifold import UMAP

def main(stdscr):
    # Setup curses terminal settings
    curses.curs_set(0) # Hide cursor
    stdscr.nodelay(True) # Non-blocking keyboard input
    stdscr.clear()

    # 1. Mocking your data (30k points, 768 dimensions)
    stdscr.addstr(0, 0, "Initializing 30,000 x 768 data on A6000 VRAM...")
    stdscr.refresh()
    X_gpu = cp.random.randn(30000, 768, dtype=cp.float32)

    # 2. Precompute the k-NN Graph once on the GPU
    stdscr.addstr(1, 0, "Precomputing 768D Nearest Neighbor Graph...")
    stdscr.refresh()
    MAX_NEIGHBORS = 100
    knn = NearestNeighbors(n_neighbors=MAX_NEIGHBORS)
    knn.fit(X_gpu)
    distances, indices = knn.kneighbors(X_gpu)

    # Core hyperparameter state
    n_neighbors = 15
    min_dist = 0.10

    # Flag to trigger recalculation on start and on keypresses
    dirty = True

    while True:
        if dirty:
            stdscr.clear()
            stdscr.addstr(0, 0, f"Recalculating UMAP layout on A6000... [Neighbors: {n_neighbors} | Min Dist: {min_dist:.2f}]")
            stdscr.refresh()

            # Slice graph instantly on GPU & optimize layout
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                knn_graph=(distances[:, :n_neighbors], indices[:, :n_neighbors]),
                init='random',
                random_state=42
            )
            embedding_gpu = umap_model.fit_transform(None)
            
            # Pull to CPU for plotext layout rendering
            embedding_cpu = embedding_gpu.get()
            x = embedding_cpu[:, 0]
            y = embedding_cpu[:, 1]

            # Clear and redraw the terminal plot window
            plt.clf()
            plt.scatter(x, y, marker="dot")
            plt.theme("dark")
            
            # Auto-size to your active terminal dimensions
            # Reserve 5 rows at the top for labels and controls
            plt.plotsize(plt.terminal_width(), plt.terminal_height() - 5)
            
            # Clear screen again to prevent artifacts, print instructions & plot
            stdscr.clear()
            stdscr.addstr(0, 0, "=== INSTANT GPU UMAP HYPERPARAMETER TUNER ===")
            stdscr.addstr(1, 0, f"Controls -> Neighbors: [1] Dec  [2] Inc ({n_neighbors})  |  Min Dist: [3] Dec  [4] Inc ({min_dist:.2f})")
            stdscr.addstr(2, 0, "Press [q] to exit.")
            
            # Inject plotext raw string directly into curses window buffers safely
            stdscr.addstr(4, 0, plt.build())
            stdscr.refresh()
            dirty = False

        # Check for non-blocking key presses
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('1'):
                n_neighbors = max(5, n_neighbors - 5)
                dirty = True
            elif key == ord('2'):
                n_neighbors = min(MAX_NEIGHBORS, n_neighbors + 5)
                dirty = True
            elif key == ord('3'):
                min_dist = max(0.01, min_dist - 0.05)
                dirty = True
            elif key == ord('4'):
                min_dist = min(0.99, min_dist + 0.05)
                dirty = True
        except Exception:
            pass

if __name__ == "__main__":
    # Wrap execution inside curses wrapper to restore terminal on exit/crash
    curses.wrapper(main)
