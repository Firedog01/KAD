import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os


def remove_old_images():
    """
    clear folder images/
    """
    filenames = os.listdir("images")
    for file in filenames:
        os.remove('images/' + file)


def save_frame(i: int, data: list, neurons: list, r: float):
    """
    saves frame for current data and neurons with name i.png
    """
    # filesize 500x500
    fig = plt.figure(figsize=(10, 10), dpi=50)

    # move axis
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout(pad=0.2)
    for row in data:
        plt.scatter(row[0], row[1], color="blue", marker=".", s=10)
    for row in neurons:
        plt.scatter(row.w[0], row.w[1], color="red", marker="o", s=25)
    plt.xlim([-r, r])
    plt.ylim([-r, r])
    path = "images/{:04d}.png".format(i)
    plt.savefig(path)
    plt.close()


def make_animation():
    """
    combines all images from images/, saves gif in output/
    """
    filenames = sorted(os.listdir("images"))
    with imageio.get_writer('output/anim.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread('images/' + filename)
            writer.append_data(image)


def make_animated_plot(data, node_states, r: float):
    """
    create animation using plt
    """
    fig, ax = plt.subplots()
    ax.set(xlim=(-r, r), ylim=(-r, r))

    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    scatr = ax.scatter([], [], color='red', s=5, marker="o")
    scatd = ax.scatter([], [], color='blue', s=2, marker=".")
    scatd.set_offsets(data)

    def animate(i):
        scatr.set_offsets(node_states[i])
        return scatr

    anim = FuncAnimation(
        fig, animate, interval=100, frames=len(node_states))

    anim.save("./output/animation.gif")
