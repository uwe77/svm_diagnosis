import numpy as np
import joblib
import os, sys
from typing import List
import cv2
import matplotlib.pyplot as plt

###############################################################################
# 1. Pixel, Known_KNN, ImageData
###############################################################################
class Pixel:
    def __init__(self, x=0, y=0, color='white', label=None):
        self.x = x
        self.y = y
        self.color = color   # "black" or "white"
        self.label = label   # cluster name, e.g. 'left_white', 'right_bubble'

    def coords(self):
        return np.array([self.x, self.y])

class Known_KNN:
    def __init__(self):
        # Seven clusters total
        self.cluster = {
            'left_white':   Pixel(x=10,  y=50, color='white'),
            'left_black':   Pixel(x=10,  y=50, color='black'),
            'right_white':  Pixel(x=90,  y=50, color='white'),
            'right_black':  Pixel(x=90,  y=50, color='black'),
            'middle_black': Pixel(x=50,  y=50, color='black'),
            'left_bubble':  Pixel(x=50,  y=50, color='black'),
            'right_bubble': Pixel(x=80,  y=50, color='black')
        }

    def predict_pixel(self, px: Pixel):
        """Return the cluster name with minimal distance among same-color clusters."""
        candidates = {}
        for c_name, c_obj in self.cluster.items():
            if c_obj.color == px.color:
                dist = np.linalg.norm(px.coords() - c_obj.coords())
                candidates[c_name] = dist
        if not candidates:
            return None
        return min(candidates, key=candidates.get)


class ImageData:
    def __init__(self, image_path, label='good', resize_dim=(100,100)):
        self.image_path = image_path
        self.label = label         # 'good' or 'bad' or 'unknown'
        self.resize_dim = resize_dim
        self.pixels: List[Pixel] = []

        self.black_threshold = 128
        self.bubble_tolerance = 10

        # We'll store the thresholded image for plotting
        self.bin_img = None

    def load_and_threshold(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: cannot load {self.image_path}")
            return
        img = cv2.resize(img, self.resize_dim)

        # create binary mask
        _, bin_img = cv2.threshold(img, self.black_threshold, 255, cv2.THRESH_BINARY)
        self.bin_img = bin_img  # store for plotting

        self.pixels.clear()
        h, w = bin_img.shape
        for y in range(h):
            for x in range(w):
                # if pixel=0 => black, else white
                color = 'black' if bin_img[y, x] == 0 else 'white'
                self.pixels.append(Pixel(x, y, color))

    def classify_image(self, known_knn: Known_KNN):
        """Assign each pixel to a cluster, count bubble pixels to decide 'good' or 'bad'."""
        bubble_count = 0
        for px in self.pixels:
            c_name = known_knn.predict_pixel(px)
            px.label = c_name
            if c_name and 'bubble' in c_name:
                bubble_count += 1

        # If bubble_count >= bubble_tolerance => 'bad'
        return 'bad' if bubble_count >= self.bubble_tolerance else 'good'

###############################################################################
# 2. Loading "good" + "bads" dataset
###############################################################################
def load_dataset(good_folder, bad_folder, resize_dim=(100,100)):
    data = []
    valid_exts = {'.jpg', '.jpeg', '.png'}

    if os.path.exists(good_folder):
        for f in os.listdir(good_folder):
            if any(f.lower().endswith(ext) for ext in valid_exts):
                path = os.path.join(good_folder, f)
                data.append(ImageData(path, label='good', resize_dim=resize_dim))

    if os.path.exists(bad_folder):
        for f in os.listdir(bad_folder):
            if any(f.lower().endswith(ext) for ext in valid_exts):
                path = os.path.join(bad_folder, f)
                data.append(ImageData(path, label='bad', resize_dim=resize_dim))

    return data

###############################################################################
# 3. PSO Implementation (with velocity clamp & freeze on position_tol)
###############################################################################
class PSO:
    """
    - velocity clamp (vel_max)
    - freeze particle if same best_score == global_best_score & positions within position_tol
    - if all frozen => stop early
    """
    def __init__(
        self,
        data: List[ImageData],
        num_particles=10,
        max_iter=10,
        w=0.5,
        c1=0.5,
        c2=0.5,
        vel_max=10.0,
        position_tol=1e-3
    ):
        self.data = data
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vel_max = vel_max
        self.position_tol = position_tol

        self.num_dimensions = 16  # (x,y)*7 + black_thr + bubble_tol
        self.global_best_score = float('inf')
        self.global_best_pos = None

        self.particles: List[Particle] = []
        self._init_particles()

    def _init_particles(self):
        for _ in range(self.num_particles):
            p = Particle(self.data, self.num_dimensions, self.vel_max)
            self.particles.append(p)

    def optimize(self):
        for iteration in range(self.max_iter):
            # Evaluate all active
            for p in self.particles:
                if not p.active:
                    continue
                score = p.evaluate()
                # local best
                if score < p.best_score:
                    p.best_score = score
                    p.best_pos = p.position.copy()
                # global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos = p.position.copy()

            # Freeze condition => same best_score & position dist < position_tol
            for p in self.particles:
                if p.active and p.best_score == self.global_best_score:
                    dist = np.linalg.norm(p.best_pos - self.global_best_pos)
                    if dist < self.position_tol:
                        p.active = False
                        p.velocity[:] = 0

            # If all inactive => stop
            all_inactive = all(not p.active for p in self.particles)
            if all_inactive:
                print(f"All particles reached global best => early stop at iteration {iteration+1}")
                break

            # Update velocity/position
            for p in self.particles:
                if p.active:
                    p.update_velocity(self.global_best_pos, self.w, self.c1, self.c2)
                    p.update_position()

            print(f"Iteration {iteration+1}/{self.max_iter}, global_best={self.global_best_score}")

        print("PSO finished.")
        return self.global_best_pos, self.global_best_score

###############################################################################
# 4. Particle
###############################################################################
class Particle:
    """
    dimension=16 => (x,y)*7 + black_threshold + bubble_tolerance
    velocity clamp => +/- vel_max
    freeze => active=False
    """
    def __init__(self, data, num_dimensions=16, vel_max=10.0):
        self.data = data
        self.num_dimensions = num_dimensions
        self.vel_max = vel_max

        self.position = np.zeros(num_dimensions)
        self.velocity = np.zeros(num_dimensions)

        self.pos_min = np.zeros(num_dimensions)
        self.pos_max = np.zeros(num_dimensions)
        # cluster coords => [0..100]
        for i in range(14):
            self.pos_min[i] = 0
            self.pos_max[i] = 100
        # black_threshold => 0..255
        self.pos_min[14] = 0
        self.pos_max[14] = 255
        # bubble_tolerance => 0..500
        self.pos_min[15] = 0
        self.pos_max[15] = 500

        # random init
        self.position = np.random.uniform(self.pos_min, self.pos_max)
        self.velocity = np.zeros(num_dimensions)

        self.best_score = float('inf')
        self.best_pos = self.position.copy()
        self.active = True

    def evaluate(self) -> float:
        """Assign each dimension => build known_knn => classify => sum misclassified."""
        known_knn = Known_KNN()
        c_names = list(known_knn.cluster.keys())
        # set cluster coords
        for idx, c_name in enumerate(c_names):
            xcoord = self.position[idx*2]
            ycoord = self.position[idx*2 + 1]
            known_knn.cluster[c_name].x = xcoord
            known_knn.cluster[c_name].y = ycoord

        black_thr = int(self.position[14])
        bubble_tol = int(self.position[15])

        misclassified = 0
        for img_data in self.data:
            img_data.black_threshold = black_thr
            img_data.bubble_tolerance = bubble_tol
            img_data.load_and_threshold()
            predicted = img_data.classify_image(known_knn)
            if predicted != img_data.label:
                misclassified += 1

        return misclassified

    def update_velocity(self, gbest_pos, w, c1, c2):
        r1 = np.random.rand(self.num_dimensions)
        r2 = np.random.rand(self.num_dimensions)
        cog = c1 * r1 * (self.best_pos - self.position)
        soc = c2 * r2 * (gbest_pos - self.position)
        self.velocity = w*self.velocity + cog + soc

        # clamp velocity
        self.velocity = np.clip(self.velocity, -self.vel_max, self.vel_max)

    def update_position(self):
        self.position += self.velocity
        # clip position
        self.position = np.maximum(self.position, self.pos_min)
        self.position = np.minimum(self.position, self.pos_max)

###############################################################################
# 5. Plotting the Original Mask + Cluster Color Image
###############################################################################
def plot_results(img_data: ImageData):
    """
    Show two subplots:
      1) The thresholded (binary) image (img_data.bin_img).
      2) A color-coded image for cluster assignments.
    """
    if img_data.bin_img is None or len(img_data.pixels) == 0:
        print("No image data to plot.")
        return

    h, w = img_data.bin_img.shape

    # 1) Convert bin_img to RGB for plotting
    bin_rgb = cv2.cvtColor(img_data.bin_img, cv2.COLOR_GRAY2RGB)

    # 2) Build cluster color image
    # define distinct colors for each cluster
    cluster_colors = {
        'left_white':   (255, 255, 255), # white
        'left_black':   (50, 50, 50),    # dark gray
        'right_white':  (200, 200, 200),
        'right_black':  (100, 100, 100),
        'middle_black': (80, 80, 80),
        'left_bubble':  (0, 0, 255),     # red
        'right_bubble': (0, 255, 0),     # green
        None:           (255, 0, 255)    # magenta for "no cluster"
    }
    cluster_img = np.zeros((h, w, 3), dtype=np.uint8)

    for px in img_data.pixels:
        color = cluster_colors.get(px.label, (255, 255, 0)) # default: cyan
        cluster_img[px.y, px.x] = color

    # show side by side
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].imshow(bin_rgb)
    axes[0].set_title("Thresholded Mask")
    axes[0].axis("off")

    axes[1].imshow(cluster_img)
    axes[1].set_title("Cluster Assignments")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

###############################################################################
# 6. TRAIN & TEST UTILS
###############################################################################
def train_kknn_with_pso(
    good_folder, bad_folder,
    resize_dim=(100,100),
    num_particles=5,
    max_iter=10,
    w=0.5,
    c1=0.4,
    c2=0.6,
    vel_max=10.0,
    position_tol=1e-3
):
    data = load_dataset(good_folder, bad_folder, resize_dim=resize_dim)
    print(f"Loaded {len(data)} images total.")
    pso = PSO(
        data=data,
        num_particles=num_particles,
        max_iter=max_iter,
        w=w,
        c1=c1,
        c2=c2,
        vel_max=vel_max,
        position_tol=position_tol
    )
    best_pos, best_score = pso.optimize()
    joblib.dump((best_pos, best_score), "kknn_particle_best.pkl")
    print(f"Saved best_pos => score={best_score}")

def test_kknn_on_new_data(test_folder, resize_dim=(100,100)):
    if not os.path.exists("kknn_particle_best.pkl"):
        print("No model found. Please train first.")
        return

    best_pos, best_score = joblib.load("kknn_particle_best.pkl")
    print(f"Loaded best model with score={best_score}")

    # Rebuild Known_KNN
    known_knn = Known_KNN()
    c_names = list(known_knn.cluster.keys())
    for idx, c_name in enumerate(c_names):
        known_knn.cluster[c_name].x = best_pos[idx*2]
        known_knn.cluster[c_name].y = best_pos[idx*2 + 1]

    black_thr = int(best_pos[14])
    bubble_tol = int(best_pos[15])
    print(f"Using black_threshold={black_thr}, bubble_tolerance={bubble_tol}")

    valid_exts = {'.jpg','.jpeg','.png'}
    files = [f for f in os.listdir(test_folder) if any(f.lower().endswith(e) for e in valid_exts)]
    for f in files:
        path = os.path.join(test_folder, f)
        img_data = ImageData(path, label='unknown', resize_dim=resize_dim)
        img_data.black_threshold = black_thr
        img_data.bubble_tolerance = bubble_tol

        img_data.load_and_threshold()
        predicted = img_data.classify_image(known_knn)
        print(f"File={f} => predicted={predicted}")

        # Now plot side-by-side
        plot_results(img_data)

def main():
    if len(sys.argv) < 2:
        print("Usage: python pso_main.py [train|test]")
        return

    cmd = sys.argv[1]
    if cmd == 'train':
        train_kknn_with_pso(
            good_folder="../images/goods",
            bad_folder="../images/bads",
            num_particles=5,
            max_iter=100,
            w=5.2,
            c1=0.4,
            c2=0.6,
            vel_max=20.0,
            position_tol=1e-3
        )
    elif cmd == 'test':
        test_kknn_on_new_data("../images/bads")
    else:
        print("Unknown command. Use train or test.")

if __name__ == "__main__":
    main()
