import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

# ১. প্রথমে ফাংশনগুলো ডিফাইন করতে হবে
def load_data(directory):
    # ট্রেনিং চ্যালেঞ্জ এবং সলিউশন লোড করা
    with open(os.path.join(directory, 'arc-agi_training_challenges.json')) as f:
        challenges = json.load(f)
    with open(os.path.join(directory, 'arc-agi_training_solutions.json')) as f:
        solutions = json.load(f)
    return challenges, solutions

def plot_task(task_id, challenges, solutions):
    task = challenges[task_id]
    # সলিউশন ফাইল থেকে ওই টাস্কের আউটপুট নেওয়া
    sol = solutions[task_id]
    
    # রঙের ম্যাপ (ARC Standard)
    cmap = colors.ListedColormap(['#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)

    num_train = len(task['train'])
    # সাবপ্লট তৈরি (২টি রো: উপরের রো-তে ইনপুট, নিচের রো-তে আউটপুট)
    fig, axs = plt.subplots(2, num_train + 1, figsize=(15, 6))

    # ট্রেনিং উদাহরণগুলো দেখানো
    for i, pair in enumerate(task['train']):
        axs[0, i].imshow(pair['input'], cmap=cmap, norm=norm)
        axs[0, i].set_axis_off()
        axs[0, i].set_title(f'Train {i} In')
        
        axs[1, i].imshow(pair['output'], cmap=cmap, norm=norm)
        axs[1, i].set_axis_off()
        axs[1, i].set_title(f'Train {i} Out')

    # টেস্ট ইনপুট এবং আসল সলিউশন দেখানো (সবচেয়ে ডানে)
    axs[0, -1].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)
    axs[0, -1].set_axis_off()
    axs[0, -1].set_title('Test Input')
    
    axs[1, -1].imshow(sol[0], cmap=cmap, norm=norm)
    axs[1, -1].set_axis_off()
    axs[1, -1].set_title('Target Solution')

    plt.tight_layout()
    plt.show()

# ২. ফাংশন ডিফাইন করার পর এখন সেগুলো কল করতে হবে
if __name__ == "__main__":
    # এটি তোমার বর্তমান ফাইলের ফোল্ডার থেকে প্রোজেক্টের মেইন ডিরেক্টরি খুঁজে নেবে
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"

    # ডেটা লোড করো
    try:
        challenges, solutions = load_data(data_dir)
        
        # প্রথম টাস্কটি দেখো
        first_task_id = list(challenges.keys())[0]
        print(f"Loading Task ID: {first_task_id}")
        plot_task(first_task_id, challenges, solutions)
        
    except FileNotFoundError:
        print(f"Error: Data folder not found at {data_dir}. Please check your folder structure.")